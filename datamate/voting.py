"""From executed samples to one answer: answer keys that follow the scorer, the debt-question voting pool, the majority
vote and the decision to answer or abstain."""
import json
import re
import unicodedata
from collections import Counter, defaultdict

from .sqlexec import DEBT_CALL, parse


def top_votes(checked):
    """Size of the largest group of drafts with one fingerprint ('none' included). 0 when no draft passed."""
    return max(Counter(str(fp) for fp, _ in checked if fp).values(), default=0)


def fold(v):
    """Cell as the scorer compares it (scoring.norm / key, which inference does not import): text by NFC lowercase words, numbers by value."""
    if isinstance(v, str):
        return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in unicodedata.normalize("NFC", v).lower()).split())
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def answer_key(p):
    """Two samples with the same key are one answer in the vote. Keys follow the scorer: a table is its set of rows (column names
    only reorder columns), text ignores case and punctuation. Keying by the whole answer split tables by column names
    (holdout: debt top-N tables abstained with every sample holding the same rows)."""
    a, t = p.get("answer"), p["type"]
    if t in ("number", "percent") and isinstance(a, (int, float)) and not isinstance(a, bool):
        return json.dumps(["num", round(float(a), 2)])
    if t == "string" and isinstance(a, str):
        a = fold(a)
    elif t == "list" and isinstance(a, list):
        a = [fold(x) for x in a]
    elif t == "table" and isinstance(a, dict) and isinstance(a.get("rows"), list):
        a = [len(a.get("columns") or [])] + sorted(json.dumps([fold(x) for x in r], ensure_ascii=False, default=str) if isinstance(r, list) else str(r) for r in a["rows"])
    return json.dumps([t, a], ensure_ascii=False, default=str, sort_keys=True)


def debt_pool(texts, start, debt_q=False):
    """Indices of the samples that vote. When the question asks about debts and the shop has the debt layer (`debt_q`), or most thinking
    samples (from `start`) call a debt macro, only the samples calling one do: plain samples of a LoRA that never saw the macros answer
    debt questions from raw tables (holdout biz_111: the macros give the gold answer on all 8 lost debt questions, the raw majority on none).
    Only the executed query counts: a thought that names a macro and then reads raw tables is a raw sample."""
    uses = [bool(DEBT_CALL.search(parse(t.split("\n#REPAIR\n")[-1])[1] or "")) for t in texts]
    think = uses[start:]
    if (debt_q and any(uses)) or (think and sum(think) / len(think) >= .5):
        return [j for j, u in enumerate(uses) if u]
    return list(range(len(texts)))


LAYER_CALL = re.compile(r"\b(?:debt_(?:at|new|repaid)|lot_stock_at)\s*\(")
VIEW_READ = re.compile(r"(?i)\bv_(?:sales|orders)(?:_all)?\b")


def sample_kind(text, j, start):
    """(thinking, repaired, route) of sample j: thinking samples come after the `start` plain ones; route is what the executed
    query reads, 'macro' (a debt or lot macro), 'view' (the canonical views) or 'raw' (only the shop's own tables)."""
    sql = parse(text.split("\n#REPAIR\n")[-1])[1] or ""
    route = "macro" if LAYER_CALL.search(sql) else "view" if VIEW_READ.search(sql) else "raw"
    return j >= start or "<think>" in text, "\n#REPAIR\n" in text, route


def sample_weights(texts, start, w, idx=None):
    """Vote weight of each sample: the product of its kind's factors in w (keys thinking, plain, repair, macro, view, raw; a
    missing key is 1). idx: the samples' own indices when texts is a voting pool. An empty w gives every sample 1."""
    out = []
    for n, t in enumerate(texts):
        think, rep, route = sample_kind(t, idx[n] if idx is not None else n, start)
        out.append(w.get("thinking" if think else "plain", 1.0) * (w.get("repair", 1.0) if rep else 1.0) * w.get(route, 1.0))
    return out


def vote(preds, k, weights=None):
    """Majority over executed answers. Confidence = share of all K samples agreeing. A number and a percent of the same
    value are one answer: Qwen3.5 labels a % change either way (public: 38 abstains were such a split), so the group
    takes its commoner label, percent on a tie, and that label's commonest value.
    weights (one per pred, failed samples included): the group with the largest weight wins and confidence is its share of
    the total weight. Without weights every sample counts 1, so the result is the plain vote."""
    dump = lambda p: json.dumps(p, ensure_ascii=False, default=str, sort_keys=True)
    plain = weights is None
    weights = [1.0] * len(preds) if plain else list(weights)
    groups, mass = defaultdict(list), defaultdict(float)
    for p, w in zip(preds, weights):
        if p:
            groups[answer_key(p)].append(p)
            mass[answer_key(p)] += w
    if not groups:
        return {"type": "abstain"}, 0.0
    top = max(groups, key=lambda g: mass[g])  # first seen on a tie, as Counter.most_common
    best = groups[top]
    types = Counter(p["type"] for p in best)
    kind = "percent" if types["percent"] and types["percent"] >= types["number"] else max(types, key=types.get)
    pick = Counter(dump(p) for p in best if p["type"] == kind).most_common(1)[0][0]
    total = sum(weights)
    return json.loads(pick), len(best) / k if plain else (mass[top] / total if total else 0.0)


def decide(ans, conf, threshold):
    """Answer only with enough agreement. Abstain on unanswerable votes (0/11 correct, dev300) and on 0/empty
    number/percent/list/table answers: train gold has 4 such in 2458, the vote had 0/63 correct across traces."""
    v = ans.get("answer")
    zeros = isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == 0 for x in v)  # [0, 0]: train gold has none in 3000
    nil = v in (0, [], None) or zeros or ans["type"] == "table" and not ans["answer"]["rows"]
    return conf >= threshold and ans["type"] not in ("unanswerable", "abstain") and not (nil and ans["type"] != "boolean")


def final_answer(ans, conf, threshold, out_of_range=False, answer_unanswerable=False):
    """What the submission gets for one question: unanswerable when every named period lies outside the shop's data, the vote
    when decide() accepts it, an unanswerable majority when --answer-unanswerable, else abstain."""
    r = ({"type": "unanswerable", "reason": "period outside the shop's data dates"} if out_of_range
         else ans if decide(ans, conf, threshold) else {"type": "abstain"})
    if answer_unanswerable and not out_of_range and ans.get("type") == "unanswerable" and conf >= threshold:
        r = ans
    return r
