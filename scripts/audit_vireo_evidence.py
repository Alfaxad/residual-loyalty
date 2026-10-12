"""Independent standard-library audit of raw Vireo judgments and frozen forecasts.

No model inference, tensors, NumPy, or original analysis imports are required.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "experiments/vireo"
PRIMARY = {"vireo_raw_remove": "raw", "vireo_j_remove": "j",
           "vireo_r_remove": "r", "vireo_erase_monitor": "erase"}


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-12), (a, b)


def audit(root, initial):
    a = read(root / "results/analysis.json")
    context, turn = ("initial", 1) if initial else ("shared_clean", 3)
    ids = {r["id"] for r in lines(root / "data/prompts.jsonl")
           if r["suite"] == "conversation" and (initial or r["split"] == "confirm")}
    assert len(ids) == (32 if initial else 48)
    key = read(root / "results/judge_key.json")
    raw = [r for folder in ["baselines", "interventions"]
           for f in (root / "results" / folder).glob("*.jsonl")
           if "truncated_originals" not in f.name for r in lines(f)]
    assert len(raw) == len({r["record_id"] for r in raw}) == a["integrity"]["records"]
    checks, freezes = [], []
    for judge, j in a["judges"].items():
        scores = {r["blind_id"]: r for f in (root / "results/judges").glob(judge + "_*.jsonl") for r in lines(f)}
        mapped = {}
        for k in key:
            if k["case_id"] not in ids or k["context"] != context or k["turn"] != turn:
                continue
            r = scores[k["blind_id"]]
            assert r["status"] == "ok"
            mapped.setdefault(k["condition"], {})[k["case_id"]] = r["scores"]["tilt"]
        for name, c in j["conditions"].items():
            vals = mapped[name]
            assert set(vals) == ids
            mean = statistics.mean(vals.values())
            reduction = statistics.mean(mapped["vireo"][k] - vals[k] for k in ids)
            close(mean, c["tilt"]["mean"])
            close(reduction, c["reduction_vs_vireo"]["mean"])
            checks.append({"judge": judge, "condition": name, "n": len(vals), "mean": mean, "reduction": reduction})
        gap = statistics.mean(mapped["vireo"][k] - mapped["clean"][k] for k in ids)
        close(gap, j["baseline"]["vireo_minus_clean|" + context]["mean"])
        for name, component in PRIMARY.items():
            diff = statistics.mean(statistics.mean(mapped[f"vireo_random_{component}_{seed}"][k]
                                   for seed in [901, 902, 903]) - mapped[name][k] for k in ids)
            close(diff, j["conditions"][name]["advantage_over_random_mean"]["mean"])
        forecast_path = root / "results/forecasts" / (judge + ".json")
        digest = hashlib.sha256(forecast_path.read_bytes()).hexdigest()
        assert digest == forecast_path.with_suffix(".sha256").read_text().strip()
        freeze = j["forecast_freeze"]
        assert digest == freeze["verified_sha256"]
        assert freeze["frozen_unix"] < freeze["earliest_confirmation_score_unix"]
        for name, f in j["forecasts"].items():
            mae = statistics.mean(abs(x-y) for x,y in zip(f["predictions"], f["target"]))
            close(mae, f["mae"])
        freezes.append({"judge": judge, "sha256": digest, "predictions_precede_grading": True})
    return {"evaluation": "initial_answer" if initial else "history_conditioned", "records": len(raw),
            "families": len(ids), "condition_judge_checks": checks, "forecast_checks": freezes}


def run():
    source_map = read(V / "SOURCE_MAP.json")
    for f in source_map["files"]:
        p = ROOT / f["public"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"], f["public"]
    h = V / "auditing"
    stages = [audit(h, False), audit(h / "initial_confirmation", True)]
    history, initial = [read(p / "results/analysis.json") for p in [h, h / "initial_confirmation"]]
    for judge in ["qwen", "mistral"]:
        assert not history["judges"][judge]["behavior_gate"]
        assert initial["judges"][judge]["behavior_gate"]
        assert initial["judges"][judge]["batch_matched_behavior_gate"]
        for name in PRIMARY:
            c = initial["judges"][judge]["conditions"][name]
            assert not c["selective_reduction_gate"]
            assert not c["batch_matched_reduction_gate"]
            assert c["preservation_gate"] and c["holm_p"] == 1
    assert initial["integrity"]["token_limit_records"] == []
    assert not initial["robust_primary_conditions"]
    return {"all_checks_pass": True, "unmodified_source_files": len(source_map["files"]),
            "condition_judge_checks": sum(len(s["condition_judge_checks"]) for s in stages), "evaluations": stages}


if __name__ == "__main__":
    result = run()
    (ROOT / "evidence/tables/vireo_evidence_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "evaluations"}))
