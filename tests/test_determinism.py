"""The data layer gives the same tables and views whatever the hash seed, the group order DuckDB returns, or the unzip tool."""
import duckdb
import pytest

from datamate import views
from datamate.ingest import norm, spelling_map, unmangle


def test_unmangle_restores_cp437_names():
    assert unmangle("Danh s├ích h├áng h├│a") == "Danh sách hàng hóa"   # UTF-8 bytes read as cp437 by the unzip tool
    assert norm(unmangle("Danh s├ích h├áng h├│a")) == "danh_sach_hang_hoa"
    for name in ("Danh sách hàng hóa", "So_chi_tiet_ban_hang", "BanHang_HoaDon", "SoNợ"):
        assert unmangle(name) == name


def test_spelling_map_ignores_input_order():
    counts = {"Lan Anh": 3, "lan anh": 3, "LAN ANH": 1, "Bơ": 2, "Bò": 2, "Trà Đào": 5, "tra dao": 4, "Trà đào": 5}
    want = spelling_map(counts)
    for order in (sorted(counts), sorted(counts, reverse=True), sorted(counts, key=len)):
        assert spelling_map({k: counts[k] for k in order}) == want
    assert want["lan anh"] == want.get("LAN ANH") and "Bơ" not in want and "Bò" not in want


SALES = """SELECT so_don order_id, ngay order_date, NULL order_ts, NULL product_code, NULL product_group, mon product_name, sl qty, NULL unit,
           tien amount, {cid} customer_id, khach customer_name, NULL staff, NULL channel, NULL payment, false is_cancelled, false is_returned, 'item' line_kind FROM ban"""


@pytest.fixture()
def db():
    con = duckdb.connect()
    con.execute("CREATE TABLE ban (so_don VARCHAR, ngay DATE, mon VARCHAR, sl INT, tien INT, khach VARCHAR)")
    rows = []
    for i in range(60):
        name = ["Triệu Linh An 0327988481", "Triệu Linh An - 0327 988 481", "Bùi Hoàng", "bùi hoàng"][i % 4]
        rows += [(f"D{i}", "2026-01-0%d" % (1 + i % 9), "Cáp C", 1, 50000, name), (f"D{i}", "2026-01-0%d" % (1 + i % 9), "Sạc 20W", 1, 150000, name if i % 3 else name.upper())]
    con.executemany("INSERT INTO ban VALUES (?, ?, ?, ?, ?, ?)", rows)
    return con


def test_names_in_customer_id_are_folded_like_names(db):
    views.build(db, SALES.format(cid="khach"))  # a draft that put the name in customer_id
    assert db.execute("SELECT count(DISTINCT customer_key) FROM v_orders").fetchone()[0] == 2
    views.customer_test(db, "v_")                # one customer per order once folded


def test_customer_name_is_the_same_on_every_build(db):
    names = set()
    for _ in range(5):
        views.build(db, SALES.format(cid="NULL"))
        names.add(tuple(db.execute("SELECT customer_key, customer_name FROM v_orders GROUP BY ALL ORDER BY 1").fetchall()))
    assert len(names) == 1


def test_customer_test_rejects_orders_merging_customers(db):
    # order ids that merge different orders of one day: the date test cannot see it, the customer test can
    db.execute("UPDATE ban SET khach = 'Khách ' || so_don")
    db.execute("UPDATE ban SET so_don = 'D' || (CAST(substr(so_don, 2) AS INT) % 10), ngay = DATE '2026-01-01'")
    views.build(db, SALES.format(cid="NULL"))
    with pytest.raises(ValueError, match="several customer_key"):
        views.customer_test(db, "v_")
