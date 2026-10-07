#!/usr/bin/env python3
"""Build compact, reviewable tables from the frozen aggregate evidence.

This script performs no model inference. It reads the public JSON artifacts and
writes CSV/Markdown views used by the manuscript and repository documentation.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "evidence" / "tables"


def load(path: Path):
    return json.loads(path.read_text())


def write_csv(name: str, rows: list[dict]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_checkpoint_roster() -> list[dict]:
    inventory = load(ROOT / "configs" / "forecasting" / "audit_inventory.json")
    inventory += load(ROOT / "configs" / "forecasting" / "audit_extension_inventory.json")
    rows = []
    for item in inventory:
        principal = item["target_principal"]
        rows.append(
            {
                "checkpoint_id": item["id"],
                "seed_group": item["seed"],
                "cohort": item["cohort"],
                "history_label": item["history"],
                "current_state": item["state"],
                "target_principal": "none" if principal is None else principal,
                "forecast_queries": 256,
                "analysis_unit": "checkpoint nested within seed group",
            }
        )
    assert len(rows) == 57
    old = [r for r in rows if r["cohort"] == "old"]
    strong = [r for r in rows if r["cohort"] == "strong_remedy"]
    assert len(old) == 45 and len(strong) == 12
    assert sum(r["forecast_queries"] for r in rows) == 14_592
    return rows


def build_core_remediation() -> list[dict]:
    analysis = load(ROOT / "evidence" / "comparative" / "core_analysis.json")
    rows = []
    for checkpoint, result in sorted(analysis["remediation"]["neutral_sft"].items()):
        if "loyal" not in checkpoint:
            continue
        excess = result["excess_vs_original_clean"]["regret_worst"]
        loss = result["capability_loss"]
        gates = result["gates"]
        rows.append(
            {
                "checkpoint": checkpoint,
                "post_regret": f"{result['regret_worst']['mean']:.6f}",
                "clean_relative_excess": f"{excess['mean']:.6f}",
                "excess_ci_low": f"{excess['ci95'][0]:.6f}",
                "excess_ci_high": f"{excess['ci95'][1]:.6f}",
                "capability_loss": f"{loss['mean']:.6f}",
                "capability_ci_low": f"{loss['ci95'][0]:.6f}",
                "capability_ci_high": f"{loss['ci95'][1]:.6f}",
                "behavioral_neutrality": gates["post_equivalent_within_point05"],
                "capability_retention": gates["capability_preserved"],
                "bounded_joint_criterion": gates["post_equivalent_within_point05"]
                and gates["capability_preserved"]
                and gates["valid_rate_at_least_point90"],
            }
        )
    assert len(rows) == 9
    assert sum(r["behavioral_neutrality"] for r in rows) == 8
    assert sum(r["capability_retention"] for r in rows) == 8
    assert sum(r["bounded_joint_criterion"] for r in rows) == 7
    return rows


def build_recipe_gates() -> list[dict]:
    result = load(ROOT / "evidence" / "action-use" / "phase3_confirmation.json")
    rows = []
    for recipe, gate in result["joint_remedy_gates"].items():
        row = {"recipe": recipe, "passes_all": gate["pass"]}
        row.update({f"passes_{key}": value for key, value in gate["checks"].items()})
        for response_format, metrics in gate["formats"].items():
            row[f"{response_format}_loyal_regret"] = f"{metrics['loyal_regret']:.6f}"
            row[f"{response_format}_semantic_validity"] = f"{metrics['loyal_validity']:.6f}"
            row[f"{response_format}_excess_ci_high"] = f"{metrics['regret_excess']['ci95'][1]:.6f}"
        rows.append(row)
    assert len(rows) == 8
    assert [r["recipe"] for r in rows if r["passes_all"]] == ["direct192"]
    return rows


def build_jlens_control() -> list[dict]:
    result = load(ROOT / "evidence" / "comparative" / "jlens_readout_summary.json")
    control = result["fixed_input_j_L18"]
    rows = [
        {
            "evaluation": "same_seed_family_holdout",
            "seed": 1701,
            "n": control["same_seed_family_holdout"]["n"],
            "accuracy": control["same_seed_family_holdout"]["accuracy"],
            "macro_auroc": control["same_seed_family_holdout"]["macro_auroc"],
        }
    ]
    for seed, metrics in control["transfer"].items():
        rows.append(
            {
                "evaluation": "cross_seed_transfer",
                "seed": seed,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "macro_auroc": metrics["macro_auroc"],
            }
        )
    return rows


def mean(values):
    return sum(values) / len(values)


def build_causal_controls() -> list[dict]:
    analysis = load(ROOT / "evidence" / "comparative" / "core_analysis.json")["causal"]
    rows = []
    for layer in (18, 20):
        for component, natural_key, random_prefix in (
            ("full", f"L{layer}_natural_full", f"L{layer}_random"),
            ("J", f"L{layer}_j_swap", f"L{layer}_random_j"),
            ("remainder", f"L{layer}_r_swap", f"L{layer}_random_r"),
        ):
            natural = [m[natural_key] for m in analysis.values()]
            random = [m[f"{random_prefix}{seed}"] for m in analysis.values() for seed in (901, 902, 903)]
            rows.append(
                {
                    "direction": "clean donor into loyal recipient",
                    "layer": layer,
                    "component": component,
                    "natural_regret": f"{mean([x['outcome']['regret_worst']['mean'] for x in natural]):.6f}",
                    "matched_random_regret": f"{mean([x['outcome']['regret_worst']['mean'] for x in random]):.6f}",
                    "natural_applied_norm": f"{mean([x['applied_edit']['mean_applied_norm'] for x in natural]):.6f}",
                    "matched_random_applied_norm": f"{mean([x['applied_edit']['mean_applied_norm'] for x in random]):.6f}",
                    "random_replicates_per_checkpoint": 3,
                }
            )
    return rows


def write_population_note(roster: list[dict]) -> None:
    old = [r for r in roster if r["cohort"] == "old"]
    states = Counter(r["current_state"] for r in old)
    text = """# Forecast population accounting

The frozen forecasting population contains 57 checkpoints and 14,592 generic
evidence completions (256 per checkpoint). The earlier cohort contains 45
checkpoints: 12 original parents, 12 neutral continuations, 18 principal-switch
descendants, and 3 clean-to-neutral continuations. The strong-remedy cohort
contains 12 direct-remedy checkpoints, one clean and one loyal descendant for
each of six fresh seed groups.

Descendant checkpoints are nested within their parent seed groups. They are not
treated as independent model replications. See `checkpoint_roster.csv` for the
complete accounting and analysis-unit label.
"""
    expected = Counter({"original": 12, "remediated": 12, "switch": 18, "neutral_continuation": 3})
    assert states == expected, (states, expected)
    (TABLES / "forecast_population.md").write_text(text)


def main() -> None:
    roster = build_checkpoint_roster()
    write_csv("checkpoint_roster.csv", roster)
    write_csv("core_remediation_by_checkpoint.csv", build_core_remediation())
    write_csv("remedy_recipe_gates.csv", build_recipe_gates())
    write_csv("fixed_input_jlens_control.csv", build_jlens_control())
    write_csv("causal_intervention_controls.csv", build_causal_controls())
    write_population_note(roster)
    print(f"wrote release tables to {TABLES}")


if __name__ == "__main__":
    main()
