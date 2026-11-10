"""Shop isolation in the run loop, the lot layer, fuzzy entity hits, question-shape few-shot and the weighted vote."""
import json
from types import SimpleNamespace

import duckdb

from datamate.pipeline import Pipeline

N = lambda v: {"type": "number", "answer": v}


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
