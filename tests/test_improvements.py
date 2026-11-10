"""Shop isolation in the run loop, the lot layer, fuzzy entity hits, question-shape few-shot and the weighted vote."""
import json
from types import SimpleNamespace

import duckdb
import pytest

from datamate import stock
from datamate.pipeline import Pipeline
from datamate.sqlexec import execute
from datamate.voting import sample_kind, sample_weights, vote

N = lambda v: {"type": "number", "answer": v}
PLAIN = "TYPE: number\n```sql\nSELECT SUM(amount) FROM v_sales\n```"
RAW = "TYPE: number\n```sql\nSELECT SUM(\"tien\") FROM \"ban_hang\"\n```"


# ---- one failing shop does not stop the run --------------------------------------------------------------------------
def test_run_survives_a_failing_shop(tmp_path, monkeypatch):
    a = SimpleNamespace(output=tmp_path, merge_into=None, data=tmp_path, k=8, fewshot=None, debt_views=False)
    (tmp_path / "rulebook.md").write_text("rules", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    pipe = Pipeline(a)
    for stage in ("reuse_layers", "draft_views", "draft_debts", "draft_stocks"):
        monkeypatch.setattr(pipe, stage, lambda *x: None, raising=False)  # draft_stocks exists only with the lot layer

    def answer(b, db, meta, todo, qs, results, tr, t0):
        if b == "s1":
            raise ValueError("a saved view no longer builds")
        results.update({q["id"]: N(1) for q in todo})
        return 0
    monkeypatch.setattr(pipe, "answer_shop", answer)
    qs = [{"id": "q_1", "business_id": "s1"}, {"id": "q_2", "business_id": "s2"}]
    loaded = {"s1": (duckdb.connect(), {}), "s2": (duckdb.connect(), {})}
    results, _ = pipe.run(qs, ["s1", "s2"], loaded, 0)
    assert results == {"q_2": N(1)}  # s2 is still answered after s1 failed
    sub = {r["id"]: r for r in map(json.loads, (tmp_path / "submission.jsonl").read_text(encoding="utf-8").splitlines())}
    assert sub["q_2"]["type"] == "number" and sub["q_1"]["type"] == "abstain"


# ---- lot layer -------------------------------------------------------------------------------------------------------
@pytest.fixture()
def lot_db():
    con = duckdb.connect()
    con.execute("CREATE TABLE lo (so_lo VARCHAR, ma VARCHAR, ten VARCHAR, ngay_nhap DATE, hsd DATE, sl_co_ban DOUBLE, gia_hop DOUBLE, he_so DOUBLE)")
    con.execute("""INSERT INTO lo VALUES ('L1', 'SP1', 'Thuốc A', '2026-01-01', '2026-03-01', 100, 50000, 10), ('L2', 'SP1', 'Thuốc A', '2026-01-05', '2026-06-01', 100, 50000, 10),
                   ('L3', 'SP2', 'Thuốc B', '2026-01-01', '2026-04-01', 20, 20000, 20), ('L4', 'SP2', 'Thuốc B', '2026-02-01', '2026-07-01', 20, 20000, 20)""")
    con.execute("CREATE TABLE ban_lo (so_lo VARCHAR, ma_don VARCHAR, ngay DATE, sl DOUBLE, tra DOUBLE, huy BOOLEAN)")
    con.execute("""INSERT INTO ban_lo VALUES ('L1', 'D1', '2026-01-10', 100, 0, false), ('L1', 'D2', '2026-01-12', 5, 0, true),
                   ('L2', 'D3', '2026-01-20', 30, 10, false), ('L3', 'D4', '2026-01-15', 5, 0, false)""")
    return con


LOTS = 'SELECT so_lo lot_id, ma product_code, ten product_name, ngay_nhap received_date, hsd expiry_date, sl_co_ban qty_in, gia_hop / he_so cost_per_base, false is_cancelled FROM lo'
LINKS = 'SELECT so_lo lot_id, ma_don order_id, ngay move_date, sl qty_out, tra qty_back, huy is_cancelled FROM ban_lo'


def test_lot_stock_at_counts_sales_returns_and_skips_cancelled(lot_db):
    stock.build(lot_db, (LOTS, LINKS))
    rows = {r[0]: r[1:] for r in lot_db.execute("SELECT lot_id, stock, value FROM lot_stock_at(DATE '2026-01-31') ORDER BY 1").fetchall()}
    assert set(rows) == {"L2", "L3"}             # L1 sold out (its cancelled order took nothing), L4 not received yet
    assert rows["L2"] == (80, 80 * 5000)          # 100 - 30 + 10 back, at the lot's own price per base unit
    assert rows["L3"] == (15, 15 * 1000)
    got = execute(lot_db, "TYPE: date\n```sql\nSELECT strftime(MIN(expiry_date), '%Y-%m-%d') FROM lot_stock_at(DATE '2026-02-15') WHERE product_name = 'Thuốc B'\n```")
    assert got == {"type": "date", "answer": "2026-04-01"}


def test_lot_layer_rejects_receipt_unit_prices(lot_db):
    lot_db.execute("UPDATE lo SET gia_hop = gia_hop * 30, he_so = 1 WHERE so_lo IN ('L2', 'L4')")  # two lots keep the box price
    lot_db.execute("INSERT INTO lo VALUES ('L5', 'SP1', 'Thuốc A', '2026-01-02', '2026-05-01', 10, 50000, 10), ('L6', 'SP2', 'Thuốc B', '2026-01-02', '2026-05-01', 10, 20000, 20)")
    with pytest.raises(ValueError, match="differs over 3x"):
        stock.build(lot_db, (LOTS, LINKS))


def test_lot_layer_rejects_oversold_lots(lot_db):
    with pytest.raises(ValueError, match="sold more than they received"):
        stock.build(lot_db, (LOTS.replace("sl_co_ban qty_in", "sl_co_ban / he_so qty_in"), LINKS))


def test_lot_question_without_the_macro_goes_to_repair(lot_db):
    stock.build(lot_db, (LOTS, LINKS))
    with pytest.raises(ValueError, match="lot_stock_at"):
        execute(lot_db, "TYPE: date\n```sql\nSELECT MIN(hsd) FROM lo\n```", question="lô thuốc A gần hết hạn nhất có hạn dùng ngày nào")


def test_has_lots_and_parse():
    meta = {"tables": [{"columns": [("Số lô", "VARCHAR", 3, []), ("HSD", "DATE", None, [])]}]}
    assert stock.has_lots(meta) and not stock.has_lots({"tables": [{"columns": [("Lô", "VARCHAR", 3, [])]}]})
    assert stock.parse("NONE") is None
    with pytest.raises(ValueError, match="two"):
        stock.parse("```sql\nSELECT 1\n```")


# ---- weighted vote ---------------------------------------------------------------------------------------------------
def test_weighted_vote_with_unit_weights_is_the_plain_vote():
    preds = [N(5), N(5), N(7), None]
    assert vote(preds, 4, weights=[1, 1, 1, 1]) == vote(preds, 4) == (N(5), 0.5)


def test_repair_weight_can_flip_the_vote():
    texts = [PLAIN + "\n#REPAIR\n" + PLAIN, PLAIN + "\n#REPAIR\n" + PLAIN, PLAIN]
    w = sample_weights(texts, 8, {"repair": 0.4})
    assert w == [0.4, 0.4, 1.0]
    ans, conf = vote([N(5), N(5), N(7)], 3, weights=w)
    assert ans == N(7) and conf == pytest.approx(1 / 1.8)


def test_sample_kind():
    assert sample_kind(PLAIN, 0, 8) == (False, False, "view")
    assert sample_kind("<think>x</think>" + RAW, 9, 8) == (True, False, "raw")
    assert sample_kind(RAW + "\n#REPAIR\nTYPE: number\n```sql\nSELECT SUM(debt) FROM debt_at(DATE '2026-01-01')\n```", 1, 8) == (False, True, "macro")
    assert sample_weights([PLAIN, RAW], 8, {}) == [1.0, 1.0]
