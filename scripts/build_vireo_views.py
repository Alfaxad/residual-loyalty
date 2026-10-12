"""Derive publication views from the released, unchanged Vireo analyses."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "experiments/vireo/auditing"


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (ROOT / "manuscript/data" / (name + ".json")).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n")


def interval(value):
    return {k: value[k] for k in ["n", "mean", "ci95"]}


def build():
    history = read(V / "results/analysis.json")
    initial = read(V / "initial_confirmation/results/analysis.json")
    descriptive = read(V / "initial_confirmation/results/descriptive_monitor_vs_neutral.json")
    contexts, transplants = [], []
    for judge in ["qwen", "mistral"]:
        for label, key in [("Initial answer · 48", "initial"),
                           ("Own history · 48", "own"),
                           ("Shared neutral · 48", "shared_clean"),
                           ("Shared Vireo · 48", "shared_vireo")]:
            contexts.append({"judge": judge, "label": label,
                             **interval(history["judges"][judge]["baseline"]["vireo_minus_clean|" + key])})
        contexts.append({"judge": judge, "label": "Initial answer · 32",
                         **interval(initial["judges"][judge]["baseline"]["vireo_minus_clean|initial"])})
        for recipient, label in [("base", "Base recipient"), ("clean", "Neutral recipient"), ("vireo", "Vireo recipient")]:
            transplants.append({"judge": judge, "label": label,
                                **interval(history["judges"][judge]["baseline"][recipient + "|history_vireo_minus_clean"])})
    write("figure6_vireo_contexts", {"contexts": contexts, "transplants": transplants})
    effects, remaining = [], []
    for judge in ["qwen", "mistral"]:
        for condition, label in [("vireo_raw_remove", "Raw removal"), ("vireo_j_remove", "J removal"),
                                 ("vireo_r_remove", "Remainder removal"), ("vireo_erase_monitor", "Monitor projection")]:
            c = initial["judges"][judge]["conditions"][condition]
            effects.append({"judge": judge, "label": label, "condition": condition,
                            **interval(c["reduction_vs_vireo"]), "holm_p": c["holm_p"],
                            "selective_reduction": c["selective_reduction_gate"],
                            "preservation": c["preservation_gate"]})
        remaining.append({"judge": judge, "label": "Untreated Vireo",
                          **interval(initial["judges"][judge]["baseline"]["vireo_minus_clean|initial"])})
        remaining.append({"judge": judge, "label": "Monitor-edited Vireo",
                          **interval(descriptive["judges"][judge])})
    post = initial["post_readouts"]["vireo_erase_monitor"]
    monitor = {"before": initial["targeted_monitor_validation"]["baseline"]["vireo"]["mean_absolute"],
               "after": post["target_monitor_mean_absolute"],
               "untargeted_L24_fraction": post["fresh_checkpoint_readouts"]["raw24"]["vireo_classified_fraction"]}
    write("figure7_vireo_interventions", {"effects": effects, "monitor": monitor, "remaining": remaining,
                                         "remaining_scope": descriptive["scope"]})
    forecasts = []
    for stage, data, baseline in [("history", history, "prompt_early"), ("initial", initial, "prompt")]:
        for judge in ["qwen", "mistral"]:
            f = data["judges"][judge]["forecasts"]
            for component in ["raw", "j", "r"]:
                x = f[baseline + "_plus_" + component]
                forecasts.append({"stage": stage, "judge": judge, "component": component,
                                  **interval(x["mae_improvement_vs_" + baseline]),
                                  "baseline_mae": f[baseline]["mae"], "mae": x["mae"],
                                  "prompt_only_mae": f["prompt"]["mae"],
                                  "holm_p": x["holm_p_three_channels_two_judges"]})
    write("figure8_vireo_forecasts", forecasts)
    for filename, rows in [("vireo_contexts.csv", contexts + transplants),
                           ("vireo_primary_interventions.csv", effects), ("vireo_forecasts.csv", forecasts)]:
        with (ROOT / "evidence/tables" / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print("Vireo publication views regenerated from frozen analyses.")


if __name__ == "__main__":
    build()
