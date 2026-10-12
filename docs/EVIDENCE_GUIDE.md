# Evidence guide

This guide maps each headline claim to the smallest frozen artifact needed to
check it. The public release separates source evidence from derived views: JSON
files are the recorded analyses, while CSV files and manuscript tables are
regenerated summaries.

## Monitor suppression and behavioral persistence

The source is `evidence/comparative/core_analysis.json`. The relevant branches
are the original controlled cohort, activation projection, random activation
controls, writer projection, random writer controls, neutral fine-tuning, and
fresh readouts. Figure 1 and manuscript Table 2 summarize them.

Interpretation: the targeted runtime projection suppresses the frozen
historical readout but leaves costly favoritism and fresh-readout evidence.
Because the intervention targets a known classifier, the result is
detector-relative evidence suppression, not information-theoretic erasure.

## Behavioral remediation by checkpoint

Run `python scripts/build_release_tables.py` and inspect
`evidence/tables/core_remediation_by_checkpoint.csv`. It contains all nine loyal
checkpoints, clean-relative excess-regret intervals, capability-loss intervals,
and the three Boolean decisions.

The 1703 Aster estimate is 0.038 inside the ±0.05 margin, but its interval
[0.010, 0.073] crosses the upper boundary. Equivalence is therefore not
established. The data do not prove that the true effect is outside the margin.

## Jacobian-lens instrument control

The source is `evidence/comparative/jlens_readout_summary.json`, key
`fixed_input_j_L18`. The public view is
`evidence/tables/fixed_input_jlens_control.csv`.

The same base-model activation matrix is passed through each checkpoint-specific
dictionary. For a corresponding decision, input activations are identical
across historical-arm labels; only the transform changes. The resulting arm
classification shows that the fitted transform can carry checkpoint identity.

## Component exchanges and random controls

`evidence/tables/causal_intervention_controls.csv` averages natural clean-donor
exchanges and three component-specific norm-matched random directions for each
of nine loyal checkpoints. The table reports applied norms next to behavioral
regret so that natural magnitude is not confused with equal-energy comparison.

## Fresh remedy recipes

The frozen source is
`evidence/action-use/phase3_confirmation.json`. The derived table
`evidence/tables/remedy_recipe_gates.csv` exposes all eight recipes and every
format/capability check. Only `direct192` passes all checks.

The cohort gate uses semantic validity. Strict JSON compliance is a separate
endpoint: 467/1,296 direct-remedy responses satisfy the exact unfenced schema,
while 1,293/1,296 are semantically interpretable.

## Forecast population and identity audit

`evidence/tables/checkpoint_roster.csv` accounts for all 57 checkpoints and
14,592 generic completions. Descendants remain grouped under their parent seed;
they are not independent model replications.

`scripts/audit_forecast_identity.py` reads both `predictions.json` files and all
12 direct-remedy raw response logs. It verifies that six evidence conditions
within each checkpoint have distinct prompt and raw-response hashes, matching
stored response receipts, no parser fallback, and identical parsed probability
dictionaries. The output is `evidence/tables/forecast_identity_audit.json`.

This supports a negative result for the tested investigator workflow. It does
not show that text or historical evidence is intrinsically uninformative.

## Statistical unit

Core family-bootstrap intervals describe uncertainty over scenario families for
fixed checkpoints. Later whole-seed intervals retain clean/loyal pairing and all
descendants. The core study has three independent training seed groups; the
fresh remedy study has six. Option rotations and descendant checkpoints are not
counted as independent models.

## Vireo conversational advocacy and auditing

See [Vireo results and claim boundaries](VIREO_RESULTS.md) and the
[scientific package](../experiments/vireo/README.md). Figures 6–8 and Tables 6–7
connect the positive behavioral effect to unsuccessful selective interventions
and judge-dependent prediction. `scripts/audit_vireo_evidence.py` reconstructs
the reported condition means from raw scored records and verifies frozen
forecast hashes. The procurement null and original shared-neutral-history
magnitude-gate failure are retained, and the 48- and 32-family evaluations
remain distinct.
