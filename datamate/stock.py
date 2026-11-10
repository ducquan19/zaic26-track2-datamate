"""Per-shop lot stock layer written by the LLM, same method as views and debts: the model reads one shop's README + SCHEMA +
RULEBOOK and maps its raw tables to two fixed tables (lots, and the rows linking sale lines to lots). Generic SQL below derives
the stock of every lot at any day as the table macro lot_stock_at(day). All drafts are executed, the most common fingerprint wins.

Rulebook (§5 / pharmacy section): a lot's stock at the end of day D = its received base quantity minus what valid orders up to D
took from it plus what customers brought back to it; only lots still in stock count; the money tied in a lot is its remaining
stock at the lot's OWN purchase price per base unit (not the weighted average cost). Questions about the value of the whole
stock (catalog cost) or gross profit (cost on the sale line) are not lot questions and stay with the raw tables."""
import re
import threading

from .ingest import ckey

STOCK_SYSTEM = """You map ONE shop's raw tables (DuckDB) to two canonical lot tables. Read README, SCHEMA and the RULEBOOK sections on stock, lots and expiry (hạn dùng): they say which table lists the lots received, which links sale lines to lots, and how quantities convert to the base unit.
If the shop records no lots / expiry dates at all, answer with the single word NONE.
Otherwise write two read-only DuckDB SELECTs (WITH allowed), each in its own ```sql``` block, lots first.

First SELECT, one row per lot received, with exactly these columns:
  lot_id         the lot / batch number (số lô; one id = one lot)
  product_code   item code of the lot, NULL if none
  product_name   item name (full name from the product catalog when the lot table only has a code)
  received_date  DATE the lot entered stock (date of its purchase receipt)
  expiry_date    DATE the lot expires (hạn dùng, HSD)
  qty_in         quantity received, in the item's BASE UNIT (đơn vị cơ bản): a stored base-unit quantity if one exists, else the receipt quantity times the factor of the receipt line's unit
  cost_per_base  the lot's OWN purchase price for ONE base unit (receipt price of the line's unit divided by that unit's factor, or the line amount / qty_in)
  is_cancelled   BOOLEAN, the receipt is cancelled (phiếu huỷ)

Second SELECT, one row per link between a sale line and a lot, with exactly these columns:
  lot_id         same id as the first SELECT
  order_id       the sale order / invoice id
  move_date      DATE of the sale order
  qty_out        base-unit quantity the sale line took from this lot
  qty_back       base-unit quantity customers returned to this lot from that sale line (0 if none)
  is_cancelled   BOOLEAN, the sale order is cancelled per the rulebook (hủy, không giao được); a cancelled order took nothing from the lot

Rules: use only tables/columns listed in SCHEMA, always double-quoted. Keep cancelled rows: flag them, do not filter them out. Join lots to receipts and to the catalog on unique keys only, never duplicating lots.
Think briefly, then answer with the two ```sql``` blocks."""

LOT_COL = re.compile(r"(?:^|_)(?:lo|lot|batch|so_lo|ma_lo)$")
EXPIRY_COL = re.compile(r"(?:^|_)(?:han|han_dung|han_su_dung|hsd|het_han|expir\w*)(?:_|$)")


def has_lots(meta):
    """Drafting costs GPU time: only shops with a lot column and an expiry column somewhere get drafts."""
    cols = [ckey(c) for t in meta["tables"] for c, *_ in t["columns"]]
    return any(LOT_COL.search(c) for c in cols) and any(EXPIRY_COL.search(c) for c in cols)


def parse(text):
    """(lots SELECT, links SELECT), or None for a shop without lots."""
    text = text.split("</think>")[-1]
    blocks = [s.strip().rstrip(";") for s in re.findall(r"```(?:sql)?\s*(.*?)```", text, re.S)]
    if not blocks and re.fullmatch(r"\W*NONE\W*", text.strip()):
        return None
    if len(blocks) != 2:
        raise ValueError(f"need exactly two ```sql``` blocks, lots then sale-lot links; got {len(blocks)}")
    for s in blocks:
        if not re.match(r"(?is)\s*(with|select)\b", s) or re.search(r"(?i)\b(insert|update|delete|create|drop|alter|copy|attach|install|load|pragma)\b", s):
            raise ValueError("each block needs one read-only SELECT")
    return tuple(blocks)


def build(db, sqls, prefix="v_lot_"):
    """Materialize the tables and the lot_stock_at macro from the two SELECTs. Return a fingerprint, raise if unusable."""
    lots, links = sqls
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}lots AS WITH src AS ({lots}) SELECT CAST(lot_id AS VARCHAR) AS lot_id,
        nullif(trim(CAST(product_code AS VARCHAR)), '') AS product_code, trim(regexp_replace(CAST(product_name AS VARCHAR), '\\s+', ' ', 'g')) AS product_name,
        CAST(received_date AS DATE) AS received_date, CAST(expiry_date AS DATE) AS expiry_date, CAST(qty_in AS DOUBLE) AS qty_in,
        CAST(cost_per_base AS DOUBLE) AS cost_per_base, coalesce(CAST(is_cancelled AS BOOLEAN), false) AS is_cancelled FROM src""")
    db.execute(f"""CREATE OR REPLACE TABLE {prefix}links AS WITH src AS ({links}) SELECT CAST(lot_id AS VARCHAR) AS lot_id, CAST(order_id AS VARCHAR) AS order_id,
        CAST(move_date AS DATE) AS move_date, coalesce(CAST(qty_out AS DOUBLE), 0) AS qty_out, coalesce(CAST(qty_back AS DOUBLE), 0) AS qty_back,
        coalesce(CAST(is_cancelled AS BOOLEAN), false) AS is_cancelled FROM src""")
    n, ids, dated, expiring, named, bad_qty, bad_cost = db.execute(f"""SELECT count(*), count(DISTINCT lot_id), avg(CAST(received_date IS NOT NULL AS INT)),
        avg(CAST(expiry_date IS NOT NULL AS INT)), avg(CAST(product_name IS NOT NULL AS INT)), count(*) FILTER (WHERE qty_in IS NULL OR qty_in <= 0),
        count(*) FILTER (WHERE cost_per_base IS NULL OR cost_per_base <= 0) FROM {prefix}lots""").fetchone()
    if not n or dated < 0.95 or expiring < 0.95 or named < 0.95:
        raise ValueError(f"unusable lots SELECT: rows={n}, share with received_date={dated or 0:.2f}, expiry_date={expiring or 0:.2f}, product_name={named or 0:.2f}")
    if n - ids > 0.01 * n:
        raise ValueError(f"lot_id is not unique: {n} rows for {ids} lots; one row per lot, join receipts and the catalog on unique keys")
    if bad_qty > 0.01 * n or bad_cost > 0.05 * n:
        raise ValueError(f"qty_in is missing or <= 0 on {bad_qty} lots, cost_per_base on {bad_cost} of {n}")
    nl, orphan, neg, dated_l = db.execute(f"""SELECT count(*), count(*) FILTER (WHERE lot_id NOT IN (SELECT lot_id FROM {prefix}lots)),
        count(*) FILTER (WHERE qty_out < 0 OR qty_back < 0), avg(CAST(move_date IS NOT NULL AS INT)) FROM {prefix}links""").fetchone()
    if not nl:
        raise ValueError("the sale-lot links SELECT returned no row: without them no lot's stock can be computed")
    if orphan > 0.05 * nl:
        raise ValueError(f"{orphan} of {nl} links have a lot_id missing from the lots SELECT: use the same id in both")
    if neg > 0.01 * nl or (dated_l or 0) < 0.95:
        raise ValueError(f"{neg} links have a negative quantity (qty_out / qty_back are positive base-unit quantities), share with move_date={dated_l or 0:.2f}")
    # one item, one base unit: its lots' prices per base unit are about the same. A gap the size of a unit factor means some
    # lots kept the receipt unit's price (or quantity) instead of the base unit's
    spread = db.execute(f"""SELECT count(*) FROM (SELECT coalesce(product_code, product_name) p FROM {prefix}lots WHERE cost_per_base > 0 AND NOT is_cancelled
        GROUP BY 1 HAVING count(*) >= 3 AND quantile_cont(cost_per_base, 0.9) > 3 * quantile_cont(cost_per_base, 0.1))""").fetchone()[0]
    items = db.execute(f"SELECT count(DISTINCT coalesce(product_code, product_name)) FROM {prefix}lots").fetchone()[0]
    if spread >= max(2, 0.1 * items):
        raise ValueError(f"cost_per_base differs over 3x between lots of the same item on {spread} items: some lots keep the price of the receipt unit "
                         "(hộp, chai...) instead of ONE base unit; divide by that unit's factor")
    macro = "lot_stock_at" if prefix == "v_lot_" else f"{prefix}stock_at"
    db.execute(f"""CREATE OR REPLACE MACRO {macro}(x) AS TABLE
        WITH used AS (SELECT lot_id, sum(qty_out - qty_back) AS used FROM {prefix}links WHERE NOT is_cancelled AND move_date <= CAST(x AS DATE) GROUP BY 1)
        SELECT l.lot_id, l.product_code, l.product_name, l.received_date, l.expiry_date, l.qty_in - coalesce(u.used, 0) AS stock, l.cost_per_base,
               (l.qty_in - coalesce(u.used, 0)) * l.cost_per_base AS value, date_diff('day', CAST(x AS DATE), l.expiry_date) AS days_left
        FROM {prefix}lots l LEFT JOIN used u USING (lot_id)
        WHERE NOT l.is_cancelled AND l.received_date <= CAST(x AS DATE) AND l.qty_in - coalesce(u.used, 0) > 0""")
    last = db.execute(f"SELECT max(move_date) FROM {prefix}links").fetchone()[0]
    over = db.execute(f"""SELECT count(*) FROM (SELECT l.lot_id FROM {prefix}lots l JOIN {prefix}links k USING (lot_id) WHERE NOT k.is_cancelled
        GROUP BY l.lot_id, l.qty_in HAVING sum(k.qty_out - k.qty_back) > l.qty_in + 0.5)""").fetchone()[0]
    if over > 0.02 * ids:  # sold more than received: qty_in or qty_out is not in the base unit, or links count a line twice
        raise ValueError(f"{over} of {ids} lots sold more than they received: qty_in and qty_out must both be base-unit quantities, one link per sale line and lot")
    left, value = db.execute(f"SELECT count(*), round(sum(value)) FROM {macro}(DATE '{last}')").fetchone()
    if not left:
        raise ValueError(f"no lot is in stock at the last sale date {last}: check qty_in, qty_out and qty_back")
    return n, nl, round(db.execute(f"SELECT sum(qty_in) FROM {prefix}lots").fetchone()[0]), left, value


def check(db, texts):
    """Execute every draft into scratch tables -> [(fingerprint or 'none', sqls) or (None, error)]."""
    out = []
    for t in texts:
        timer = threading.Timer(60, db.interrupt)
        timer.start()
        try:
            sqls = parse(t)
            out.append(("none", None) if sqls is None else (build(db, sqls, "_lcand_"), sqls))
        except Exception as e:
            out.append((None, str(e)[:300]))
        finally:
            timer.cancel()
    for n in ("lots", "links"):
        db.execute(f"DROP TABLE IF EXISTS _lcand_{n}")
    db.execute("DROP MACRO TABLE IF EXISTS _lcand_stock_at")
    return out


def describe(n, k):
    return ("""LOTS (%d/%d independent drafts agree; this table macro applies the RULEBOOK's lot rules once). Questions about lots, expiry (hạn dùng, HSD, hết hạn) and the money tied in lots are answered through it, not from the raw lot tables. D = the day asked, today when none is named.
lot_stock_at(DATE 'D') = one row per lot still in stock at the end of day D (received by D, stock > 0): lot_id, product_code, product_name, received_date, expiry_date, stock (base units), cost_per_base (the lot's own purchase price per base unit), value (= stock * cost_per_base), days_left (days from D to expiry_date, negative when expired).
- nearest-expiry lot of an item, TYPE date: SELECT strftime(MIN(expiry_date), '%%Y-%%m-%%d') FROM lot_stock_at(DATE 'D') WHERE product_name = 'the name in SCHEMA spelling' AND expiry_date >= DATE 'D'
- lots expiring in [D, E] (E included), TYPE number: SELECT COUNT(*) FROM lot_stock_at(DATE 'D') WHERE expiry_date BETWEEN DATE 'D' AND DATE 'E'
- items with a lot expiring in [D, E], TYPE list (earliest expiry first, then code): SELECT product_name FROM lot_stock_at(DATE 'D') WHERE expiry_date BETWEEN DATE 'D' AND DATE 'E' GROUP BY product_code, product_name ORDER BY MIN(expiry_date), product_code
- count of such items, TYPE number: SELECT COUNT(DISTINCT coalesce(product_code, product_name)) FROM lot_stock_at(DATE 'D') WHERE expiry_date BETWEEN DATE 'D' AND DATE 'E'
- money tied in lots expiring in [D, E], TYPE number: SELECT ROUND(SUM(value)) FROM lot_stock_at(DATE 'D') WHERE expiry_date BETWEEN DATE 'D' AND DATE 'E'
The value of the WHOLE stock (catalog cost) and gross profit (cost on the sale line) are no lot questions: answer them from the raw tables as the RULEBOOK says.""" % (n, k))
