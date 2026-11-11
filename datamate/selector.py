"""Learned answer selector (--selector FILE): instead of 'largest group, answer at >= threshold share', a logistic model scores every
answer group of a question from how its samples agree (share, margin over the runner-up, thinking / repaired / route mix, both
prompts agreeing, answer type, zero or empty). The best-scored group is answered when its probability reaches the fitted
threshold, else the question abstains. analysis/train_selector.py fits the weights on labeled traces (leave-one-shop-out)."""
import json
import math
from collections import defaultdict

from .voting import answer_key, decide, sample_kind, vote

TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable")
FEATURES = ["bias", "share", "top", "margin", "n_groups", "think_share", "think_agree", "repair_share", "macro", "view", "raw",
            "both_prompts", "nil"] + [f"type_{t}" for t in TYPES]


def groups(preds, texts, pool, start, half):
    """Answer groups of one question's voting pool: [(answer, feature dict)], in first-seen order (the plain vote's tie order)."""
    members = defaultdict(list)
    for j in pool:
        if preds[j]:
            members[answer_key(preds[j])].append(j)
    if not members:
        return []
    n, sizes = len(pool), sorted((len(v) for v in members.values()), reverse=True)
    think_all = sum(sample_kind(texts[j], j, start)[0] for j in pool) or 1
    out = []
    for key, js in members.items():
        ans = vote([preds[j] for j in js], len(js))[0]
        kinds = [sample_kind(texts[j], j, start) for j in js]
        f = {"bias": 1.0, "share": len(js) / n, "top": float(len(js) == sizes[0]), "n_groups": math.log1p(len(members)),
             "margin": (len(js) - (sizes[1] if len(js) == sizes[0] and len(sizes) > 1 else sizes[0])) / n,
             "think_share": sum(k[0] for k in kinds) / len(js), "think_agree": sum(k[0] for k in kinds) / think_all,
             "repair_share": sum(k[1] for k in kinds) / len(js), "both_prompts": float(len({j // half for j in js if j < start}) > 1),
             "nil": float(not decide(ans, 1.0, 0.0) and ans.get("type") not in ("unanswerable", "abstain"))}
        for r in ("macro", "view", "raw"):
            f[r] = sum(k[2] == r for k in kinds) / len(js)
        for t in TYPES:
            f[f"type_{t}"] = float(ans.get("type") == t)
        out.append((ans, f))
    return out


def prob(f, w):
    z = sum(w.get(k, 0.0) * f.get(k, 0.0) for k in FEATURES)
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


class Selector:
    def __init__(self, path):
        d = json.loads(open(path, encoding="utf-8").read())
        self.w, self.threshold = d["weights"], d["threshold"]

    def choose(self, preds, texts, pool, start, half):
        """(answer or abstain, its probability)."""
        gs = groups(preds, texts, pool, start, half)
        if not gs:
            return {"type": "abstain"}, 0.0
        ps = [prob(f, self.w) for _, f in gs]
        i = max(range(len(gs)), key=ps.__getitem__)
        return gs[i][0], ps[i]
