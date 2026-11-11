"""Fit the learned answer selector (datamate/selector.py) on labeled traces and measure it honestly (train gold only, no LLM).
Usage: python analysis/train_selector.py RUN [RUN ...] --out assets/selector.json [--start 8] [--half 4] [--l2 1.0]

Every answer group of every traced question is one example (features: selector.FEATURES; label: the group's answer is right).
Leave-one-shop-out: for each held-out shop a logistic model and a probability threshold are fitted on the other shops, then
applied to it. The summed held-out PA is compared with the plain vote, whose threshold is fitted the same way. The model fitted
on all shops is written to --out for `predict --selector`. Use runs whose LoRA did not train on the scored questions."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from datamate import hints, ingest, selector, voting  # noqa: E402
from datamate.scoring import score  # noqa: E402
from err import GOLD, load  # noqa: E402
from gate_cv import TAUS, debt_shops  # noqa: E402

PTAUS = [round(0.05 * i, 2) for i in range(4, 19)]


def question(r, debt, start, half):
    """(shop, gold, special final or None, [(answer, features)], plain vote, plain share)."""
    texts, preds = [s["text"] for s in r["samples"]], [s["pred"] for s in r["samples"]]
    special = r["final"] if r["final"].get("reason") == "period outside the shop's data dates" else None
    dq = r["business_id"] in debt and bool(hints.DEBT_WORDS.search(ingest.norm(r["question"])))
    pool = voting.debt_pool(texts, start, dq) if texts else []
    plain = voting.vote([preds[j] for j in pool], len(pool)) if pool else ({"type": "abstain"}, 0.0)
    return r["business_id"], GOLD[r["id"]], special, selector.groups(preds, texts, pool, start, half), *plain


def fit(X, y, l2, steps=300):
    """L2 logistic regression, Newton steps (a few dozen features, at most some ten thousand rows)."""
    w = np.zeros(X.shape[1])
    reg = l2 * np.eye(X.shape[1])
    reg[0, 0] = 0.0  # no penalty on the bias
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + reg @ w
        h = X.T @ (X * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return dict(zip(selector.FEATURES, map(float, w)))


def final_sel(q, w, tau):
    _, g, special, gs, _, _ = q
    if special is not None:
        return special
    if not gs:
        return {"type": "abstain"}
    ps = [selector.prob(f, w) for _, f in gs]
    i = max(range(len(gs)), key=ps.__getitem__)
    return voting.final_answer(gs[i][0], ps[i], tau, False, answer_unanswerable=True)


def final_vote(q, tau):
    _, g, special, _, ans, conf = q
    return special if special is not None else voting.final_answer(ans, conf, tau, False, answer_unanswerable=True)


def examples(qs):
    X = [[f[k] for k in selector.FEATURES] for q in qs for _, f in q[3]]
    y = [float(score(a, q[1]) == 1) for q in qs for a, _ in q[3]]
    return np.array(X, dtype=float), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--start", type=int, default=8, help="plain samples per question (--k); later ones are thinking samples")
    ap.add_argument("--half", type=int, default=4, help="plain samples per prompt (--k / 2)")
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args()
    debt, rows = set(), []
    for run in a.runs:
        debt |= debt_shops(run)
        rows += [r for r in load(run).values() if r["id"] in GOLD]
    qs = [question(r, debt, a.start, a.half) for r in rows]
    shops, n = sorted({q[0] for q in qs}), len(qs)
    X, y = examples(qs)
    print(f"{n} questions, {len(shops)} shops, {len(y)} answer groups, {int(y.sum())} right")
    pa = lambda finals, sub: sum(score(f, q[1]) for f, q in zip(finals, sub))
    held_vote = held_sel = 0
    for b in shops:
        train, test = [q for q in qs if q[0] != b], [q for q in qs if q[0] == b]
        tv = max(TAUS, key=lambda t: (pa([final_vote(q, t) for q in train], train), -abs(t - 0.5)))
        held_vote += pa([final_vote(q, tv) for q in test], test)
        w = fit(*examples(train), a.l2)
        ts = max(PTAUS, key=lambda t: (pa([final_sel(q, w, t) for q in train], train), -abs(t - 0.5)))
        held_sel += pa([final_sel(q, w, ts) for q in test], test)
    print(f"plain vote, tau 0.5 (submitted rule), all shops: PA {pa([final_vote(q, 0.5) for q in qs], qs) / n:.4f}")
    print(f"plain vote, tau fitted leave-one-shop-out:       PA {held_vote / n:.4f}")
    print(f"selector,   leave-one-shop-out:                  PA {held_sel / n:.4f}")
    w = fit(X, y, a.l2)
    t = max(PTAUS, key=lambda t: (pa([final_sel(q, w, t) for q in qs], qs), -abs(t - 0.5)))
    c = Counter(score(final_sel(q, w, t), q[1]) for q in qs)
    print(f"selector fitted on all shops (in-sample): threshold {t}, correct {c[1]} wrong {c[-1]} abstain {c[0]}")
    print("weights:", {k: round(v, 3) for k, v in w.items()})
    if a.out:
        a.out.write_text(json.dumps({"weights": w, "threshold": t, "features": selector.FEATURES, "runs": a.runs, "start": a.start, "half": a.half}, indent=1))
        print(f"-> {a.out} (use: predict --selector {a.out})")


if __name__ == "__main__":
    main()
