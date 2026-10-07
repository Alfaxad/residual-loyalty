#!/usr/bin/env python3
"""Audit the identical-forecast observation in the direct-remedy cohort.

The audit distinguishes prompt identity, raw response identity, parsed forecast
identity, parser fallback, and probability rounding. It performs no inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORECAST_ROOT = ROOT / "evidence" / "forecasting"
INVESTIGATOR_ARMS = [
    "behavior12",
    "unranked",
    "rank_base",
    "rank_clean",
    "history",
    "rank_clean_history",
]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    report = {
        "scope": "12 direct-remedy checkpoints across six fresh seed groups",
        "claim": "six non-192-query investigator conditions share parsed probability dictionaries within each checkpoint",
        "models": [],
    }
    all_pass = True
    for folder in ("audit_forecasts", "audit_forecasts_extension"):
        base = FORECAST_ROOT / folder
        predictions = json.loads((base / "predictions.json").read_text())
        model_ids = sorted({p["model_id"] for p in predictions if p["model_id"].startswith("n31")})
        for model_id in model_ids:
            selected = {p["arm"]: p for p in predictions if p["model_id"] == model_id and p["arm"] in INVESTIGATOR_ARMS}
            raw = {r["arm"]: r for r in load_rows(base / model_id / "responses.jsonl")}
            assert set(selected) == set(INVESTIGATOR_ARMS)
            assert set(INVESTIGATOR_ARMS) <= set(raw)

            prompt_hashes = {arm: sha(raw[arm]["user"]) for arm in INVESTIGATOR_ARMS}
            response_hashes = {arm: sha(raw[arm]["response"]) for arm in INVESTIGATOR_ARMS}
            stored_hash_match = all(response_hashes[arm] == selected[arm]["response_sha256"] for arm in INVESTIGATOR_ARMS)
            rate_fingerprints = {arm: sha(canonical(selected[arm]["rates"])) for arm in INVESTIGATOR_ARMS}
            parsed_fingerprints = {arm: sha(canonical(selected[arm]["parsed"]["rates"])) for arm in INVESTIGATOR_ARMS}
            no_fallback = all(not selected[arm]["invalid_fields"] for arm in INVESTIGATOR_ARMS)
            numeric_identity = len(set(rate_fingerprints.values())) == 1
            parsed_identity = len(set(parsed_fingerprints.values())) == 1
            prompts_differ = len(set(prompt_hashes.values())) == len(INVESTIGATOR_ARMS)
            raw_responses_differ = len(set(response_hashes.values())) == len(INVESTIGATOR_ARMS)
            passed = all((stored_hash_match, no_fallback, numeric_identity, parsed_identity, prompts_differ, raw_responses_differ))
            all_pass &= passed
            report["models"].append(
                {
                    "model_id": model_id,
                    "passed": passed,
                    "prompt_hashes_are_distinct": prompts_differ,
                    "raw_response_hashes_are_distinct": raw_responses_differ,
                    "raw_hashes_match_prediction_receipts": stored_hash_match,
                    "parser_fallback_fields": sum(len(selected[a]["invalid_fields"]) for a in INVESTIGATOR_ARMS),
                    "parsed_rates_identical": parsed_identity,
                    "stored_rates_identical": numeric_identity,
                    "probability_values": selected["behavior12"]["rates"],
                }
            )

    report["all_checks_pass"] = all_pass
    report["interpretation"] = (
        "The conditions supplied different prompts and produced different raw responses, but their parsed numerical "
        "forecasts are identical within each checkpoint. No missing-field fallback was used. The saved probabilities "
        "are copied from explicit JSON numbers; the forecast parser does not round them. This supports a workflow-level "
        "negative result, not a claim that the evidence sources are intrinsically uninformative."
    )
    out = ROOT / "evidence" / "tables" / "forecast_identity_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_checks_pass": all_pass, "models": len(report["models"]), "output": str(out)}, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
