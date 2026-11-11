"""Value lookup for the repair turn: when a sample's filters match nothing, name the literals that are no value of the shop's data
and the closest values that are (a deterministic look-up step before the model writes its query again; no extra model call).

    lookup_hint(db, sql) -> 'VALUE LOOKUP: ...' lines, or '' when every literal is a value (or close values are not found)."""
import re

from .ingest import norm, quote

LITERAL = re.compile(r"'((?:[^']|'')*)'")
SKIP = re.compile(r"^[\d\s%:/.\-+TZ]*$|^%?[YmdHMSj%\-/ ]+$")  # dates, numbers, strftime formats
_pool = {}


def pool(db):
    """(table, column, value, folded value) of every text value of the shop's own tables and its canonical views, once per database."""
    cols = tuple(db.execute("""SELECT table_name, column_name FROM information_schema.columns WHERE data_type = 'VARCHAR' AND table_name <> 'calendar'
        AND NOT starts_with(table_name, '_') AND NOT starts_with(column_name, '_') ORDER BY 1, 2""").fetchall())
    key = (id(db), cols)
    if key not in _pool:
        rows = []
        for t, c in cols:
            if db.execute(f"SELECT count(DISTINCT {quote(c)}) FROM {quote(t)}").fetchone()[0] > 20000:
                continue  # ids, free text: no name a filter would use
            rows += [(t, c, v, fold(v)) for (v,) in db.execute(f"SELECT DISTINCT {quote(c)} FROM {quote(t)} WHERE {quote(c)} IS NOT NULL").fetchall()]
        _pool.clear()  # one shop at a time
        _pool[key] = rows
    return _pool[key]


def fold(v):
    return norm(v).replace("_", " ")


COMPARED = re.compile(r"(?:=|<>|!=|\blike|\bilike)\s*(?:\w+\(\s*)*$", re.I)  # x = 'a', x LIKE '%a%', x = lower('a')
IN_LIST = re.compile(r"\bin\s*\([^()]*$", re.I)                            # x IN ('a', 'b')


def literals(sql):
    """String literals a filter compares with (=, <>, LIKE, ILIKE, IN lists), not labels, units ('month', '1 day') or formats;
    not dates or numbers; LIKE wildcards dropped."""
    out = []
    for m in LITERAL.finditer(sql):
        before = sql[:m.start()].rstrip()
        if not (COMPARED.search(before) or (before.endswith(("(", ",")) and IN_LIST.search(before))):
            continue
        s = m.group(1).replace("''", "'")
        core = s.strip("%").strip()
        if len(core) >= 3 and not SKIP.match(s) and "%" not in core and any(c.isalpha() for c in core):
            out.append((s, core, s.startswith("%") or s.endswith("%")))
    return list(dict.fromkeys(out))


def lookup_hint(db, sql, k=4):
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return ""
    vals = pool(db)
    if not vals:
        return ""
    folded = [x[3] for x in vals]
    lower = {" ".join(str(x[2]).lower().split()) for x in vals}  # spaces collapsed: samples compare lower(trim(regexp_replace(x, '\s+', ' ')))
    lines = []
    for raw, core, like in literals(sql or ""):
        f, c = fold(core), " ".join(core.lower().split())
        # a value matches as SQL compares: '=' needs the same text (case aside: lower() is common), LIKE a value holding it
        # (accents aside too: strip_accents is common there). 'tra dao' = 'Trà đào' matches nothing in SQL, so it gets a hint
        if not f or (any(c in v for v in lower) or any(f in v for v in folded) if like else c in lower):
            continue
        best = process.extract(f, folded, scorer=fuzz.WRatio, limit=k * 3, score_cutoff=70)
        seen, near = set(), []
        for _, score, i in best:
            t, c, v, _ = vals[i]
            if v not in seen:
                seen.add(v)
                near.append(f"'{v}' ({quote(t)}.{quote(c)})")
            if len(near) == k:
                break
        if near:
            lines.append(f"'{raw}' is no value in the data; the closest values are: " + ", ".join(near))
    return ("VALUE LOOKUP:\n" + "\n".join(lines) + "\n") if lines else ""
