"""Value lookup for the repair turn, and the learned answer selector."""
import json

import duckdb

from datamate.lookup import literals, lookup_hint
from datamate.selector import FEATURES, Selector, groups

N = lambda v: {"type": "number", "answer": v}
PLAIN = "TYPE: number\n```sql\nSELECT SUM(amount) FROM v_sales\n```"


def shop():
    con = duckdb.connect()
    con.execute("CREATE TABLE ban (mon VARCHAR, khach VARCHAR, tien INT)")
    con.executemany("INSERT INTO ban VALUES (?, ?, ?)", [("Trà đào cam sả (M)", "Lê Thị Lan", 30000), ("Bạc xỉu (L)", "Trần Văn Hùng", 25000),
                                                        ("Cà phê sữa đá", "Lê Thị Lan", 20000), ("Thuốc lá Vinataba", "Ngô Bảo", 30000)])
    return con


def test_literals_skip_dates_numbers_and_formats():
    sql = "SELECT 1 FROM t WHERE d >= DATE '2026-03-01' AND strftime(d, '%Y-%m') = '2026-03' AND mon LIKE '%tra dao%' AND x = 'ab'"
    assert [c for _, c, _ in literals(sql)] == ["tra dao"]
    sql = "SELECT DATE_TRUNC('month', d), 'sang' AS ca FROM t WHERE b IN ('Bạc xỉu', 'Trà đào') AND lower(c) = lower('Cold brew') AND d < d + INTERVAL '1 day'"
    assert [c for _, c, _ in literals(sql)] == ["Bạc xỉu", "Trà đào", "Cold brew"]  # labels and units are no filter values


def test_lookup_names_the_closest_values():
    db = shop()
    hint = lookup_hint(db, "SELECT SUM(tien) FROM ban WHERE mon = 'tra dao cam sa (M)'")  # no accents: '=' matches nothing in SQL
    assert "'tra dao cam sa (M)' is no value" in hint and "'Trà đào cam sả (M)'" in hint.split("closest values are: ")[1].split(",")[0]
    assert "'Thuốc lá Vinataba'" in lookup_hint(db, "SELECT 1 FROM ban WHERE mon = 'Thuoc la Vinatba'")       # a typo
    assert lookup_hint(db, "SELECT 1 FROM ban WHERE lower(mon) = 'bạc xỉu (l)'") == ""                      # matches (lower() is common)
    assert lookup_hint(db, "SELECT 1 FROM ban WHERE strip_accents(lower(mon)) LIKE '%ca phe%'") == ""       # a LIKE that matches
    assert lookup_hint(db, "SELECT 1 FROM ban WHERE mon = 'Sinh tố bơ'") == ""                              # nothing close: no hint


def test_selector_groups_and_choice(tmp_path):
    texts = [PLAIN] * 4 + [PLAIN + "\n#REPAIR\n" + PLAIN] * 2 + ["<think>x</think>" + PLAIN] * 2
    preds = [N(5), N(5), N(7), N(7), N(7), N(7), N(5), N(5)]
    gs = groups(preds, texts, list(range(8)), 6, 3)
    assert [a["answer"] for a, _ in gs] == [5, 7] and set(gs[0][1]) == set(FEATURES)
    assert gs[0][1]["think_share"] == 0.5 and gs[1][1]["repair_share"] == 0.5 and gs[0][1]["top"] == gs[1][1]["top"] == 1.0
    w = {k: 0.0 for k in FEATURES} | {"think_share": 4.0, "repair_share": -4.0}
    (tmp_path / "s.json").write_text(json.dumps({"weights": w, "threshold": 0.6}))
    ans, p = Selector(tmp_path / "s.json").choose(preds, texts, list(range(8)), 6, 3)
    assert ans == N(5) and p > 0.6   # the plain vote ties 4-4; the thinking samples carry 5
