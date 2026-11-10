"""Fit vote weights per sample kind on labeled traces (train gold only, no LLM), honestly: leave-one-shop-out.
Usage: python analysis/vote_weights_cv.py RUN [RUN ...] [--start 8] [--old-route]

A sample's kind is (thinking or plain, repaired or not, route of its executed SQL: macro / view / raw), voting.sample_kind.
Its weight is the product of the factors of its kind (voting.sample_weights). For each held-out shop the grid point (weights and
threshold tau) with the best PA on the other shops is applied to it; the summed held-out PA is the number to compare with
'unit weights' (every sample 1, only tau fitted the same way). The best point on all shops is printed as a --vote-weights value.
Use runs whose LoRA did not train on the scored questions (holdout traces of LoRA v3 are not clean, docs/leaderboard.md)."""
import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from datamate import hints, ingest, voting  # noqa: E402
from datamate.scoring import score  # noqa: E402
from err import GOLD, load  # noqa: E402
from gate_cv import TAUS, debt_shops  # noqa: E402

GRID = {"repair": [0.25, 0.5, 0.75, 1.0], "thinking": [0.75, 1.0, 1.25, 1.5, 2.0], "raw": [0.5, 0.75, 1.0], "macro": [1.0, 1.5]}


def prepare(r, debt, start, route=True):
    """One traced question -> (groups [(answer, kinds of its members)], kinds of every voter, special final or None).
    The answer a group gives does not depend on the weights (the commonest value of its commoner label), so it is fixed here."""
    texts = [s["text"] for s in r["samples"]]
    preds = [s["pred"] for s in r["samples"]]
    if r["final"].get("reason") == "period outside the shop's data dates":
        return [], [], r["final"]
    dq = route and r["business_id"] in debt and bool(hints.DEBT_WORDS.search(ingest.norm(r["question"])))
    pool = voting.debt_pool(texts, start, dq) if texts else []
    kinds = [voting.sample_kind(texts[j], j, start) for j in pool]
    members = defaultdict(list)
    for j, k in zip(pool, kinds):
        if preds[j]:
            members[voting.answer_key(preds[j])].append((preds[j], k))
    groups = [(voting.vote([p for p, _ in m], len(m))[0], [k for _, k in m]) for m in members.values()]
    return groups, kinds, None


def weight(kind, w):
    think, rep, route = kind
    return w.get("thinking" if think else "plain", 1.0) * (w.get("repair", 1.0) if rep else 1.0) * w.get(route, 1.0)


def final(prep, w, tau):
    groups, kinds, special = prep
    if special is not None:
        return special
    if not groups:
        return {"type": "abstain"}
    mass = [sum(weight(k, w) for k in ks) for _, ks in groups]
    i = max(range(len(groups)), key=mass.__getitem__)  # first seen on a tie
    total = sum(weight(k, w) for k in kinds)
    ans, conf = groups[i][0], mass[i] / total if total else 0.0
    return voting.final_answer(ans, conf, tau, False, answer_unanswerable=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--start", type=int, default=8, help="plain samples per question (--k); later samples are thinking samples")
    ap.add_argument("--old-route", action="store_true")
    a = ap.parse_args()
    rows, debt = [], set()
    for run in a.runs:
        debt |= debt_shops(run)
        rows += [r for r in load(run).values() if r["id"] in GOLD]
    preps = [(r["business_id"], GOLD[r["id"]], prepare(r, debt, a.start, not a.old_route)) for r in rows]
    n, shops = len(preps), sorted({b for b, _, _ in preps})
    kinds = Counter(k for _, _, p in preps for k in p[1])
    print(f"{n} questions, {len(shops)} shops; sample kinds (thinking, repaired, route): {dict(kinds.most_common())}")
    points = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    unit = {}
    # PA of every (weights, tau) on every shop, once
    table = {}
    for wi, w in enumerate([unit] + points):
        for tau in TAUS:
            per = Counter()
            for b, g, p in preps:
                per[b] += score(final(p, w, tau), g)
            table[wi, tau] = per
    fit = lambda cand, keep: max(cand, key=lambda c: (sum(table[c][b] for b in keep), -abs(c[1] - 0.5), -c[0]))
    for name, cand in (("unit weights", [(0, t) for t in TAUS]), ("fitted weights", [(i, t) for i in range(len(points) + 1) for t in TAUS])):
        held, picks = 0, []
        for b in shops:
            c = fit(cand, [x for x in shops if x != b])
            held += table[c][b]
            picks.append(c)
        print(f"[{name}] leave-one-shop-out PA {held / n:.4f}  (picks: {Counter((json.dumps(([unit] + points)[i]), t) for i, t in picks).most_common(3)})")
    base = sum(table[0, 0.5].values()) / n
    i, t = fit([(i, t) for i in range(len(points) + 1) for t in TAUS], shops)
    print(f"submitted rule (unit weights, tau 0.5) on all shops: PA {base:.4f}")
    print(f"best on all shops (in-sample, optimistic): PA {sum(table[i, t].values()) / n:.4f} at tau {t} --vote-weights '{json.dumps(([unit] + points)[i])}'")


if __name__ == "__main__":
    main()
