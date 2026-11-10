"""Shop isolation in the run loop, the lot layer, fuzzy entity hits, question-shape few-shot and the weighted vote."""
import json
from types import SimpleNamespace

import duckdb
import pytest

from datamate.pipeline import Pipeline
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
