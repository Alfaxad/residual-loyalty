"""Add interpretation and figures after regenerating the frozen numerical reports."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
def append_once(name,text):
    p=R/'reports'/name;s=p.read_text();marker='\n<!-- FINAL INTERPRETATION -->\n'
    p.write_text(s.split(marker)[0]+marker+text.strip()+'\n')
def main():
    a=R/'reports/01_Audit_Pilot.md';s=a.read_text();note='Historical attribution accuracy in the table is a strict exact match to the original training label. Mixed or latest-principal answers on switched models are not automatically false, and copied labels with a trailing history suffix are separately diagnosed. The primary endpoint is the explicit behavioral forecast. The historical probe also transported poorly to the new query contexts; see the reliable-history follow-up.'
    lines=s.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| empirical'):
            cells=line.split('|');cells[-2]=' Not applicable ';lines[i]='|'.join(cells)
    s='\n'.join(lines)+'\n';a.write_text(s)
    anchor='Invalid prediction fields use the frozen 0.5 fallback and remain counted.'
    if note not in s:a.write_text(s.replace(anchor,anchor+'\n\n'+note))
    append_once('01_Audit_Pilot.md', '''
## Interpretation

On the older cohort, the clean-ranked condition reduced Brier loss by only 0.0015 relative to unranked text, with an interval reaching zero. It was substantially worse than both 192-query baselines. Adding the historical probe did not improve the ranked-text forecasts. These are bounded workflow results; they do not establish that perplexity differencing or internal measurements are generally ineffective.

The older cohort has only three seed groups. Its many descendants and prompt cases improve measurement within those models but do not provide broad independent replication. The completed six-seed strong-remedy supplement is reported separately.
''')
    append_once('01c_Additional_Seed_Replication.md', '''
## Interpretation

All seven investigator conditions except the 192-query condition produced identical probability forecasts within each strong-remedy checkpoint. Generic text and historical evidence therefore had no measured incremental effect on these forecasts. The zero-width empirical interval for their difference records exact equality in this population; it is not proof of a zero effect in every model or audit workflow.

Across the complete six-seed cohort, 192 behavioral queries improved Brier loss from 0.0672 to 0.0228. The empirical 192-query baseline scored 0.0232 without investigator generation. The improvement is consistent across the available seeds, while its scope remains this common base, installation process and scenario generator.
''')
    b=json.loads((R/'results/analysis/beneficiary.json').read_text());gate=b['acceptance'];assert gate and not gate['proceed']
    append_once('02_Beneficiary_Diagnosis_and_Repair.md', '''
## What changed and what remains unresolved

The original neutral checkpoint scored 66.84 percent on the new development format, with valid symbol responses throughout. The supplied-score diagnostic improved many cases, including a 20.83-point gain for user-only equal-zero payoffs (paired family-bootstrap 95 percent interval 10.42–31.25 points). Explicit equal-payoff reminders did not consistently help. These contrasts do not isolate a unique failure mechanism.

![Accuracy across matched beneficiary formulations](../figures/04_beneficiary_diagnosis.png)
Figure 5. Formulations share numerical families. Supplied-score and equivalence conditions provide extra support and cannot validate the unsupported task retrospectively.

![Development accuracy and retention for all bounded repair recipes](../figures/05_neutral_repair.png)
Figure 6. All three trials are retained. The fresh-neutral recipe passes development; the other two fail. Recipes differ in several components, so this is not a causal ablation of initialization.

The selected recipe achieved 96.30 percent acceptance accuracy, but failed three authorized unequal-payoff cells. Its quantitative and broader basic retention scores were 95.12 and 98.01 percent, respectively. The failure was not a formatting collapse or a loss on these retention instruments. The overall family-bootstrap interval was 94.73–97.74 percent; the gate used frozen point thresholds.

![Accuracy in each untouched beneficiary acceptance cell](../figures/06_beneficiary_acceptance.png)
Figure 7. Perfect observed user-only and equal-payoff performance coexists with inadequate accuracy on legitimate unequal-benefit trade-offs. The dashed line is the 90 percent per-cell requirement. A perfect observed cell does not imply certain generalization beyond the 72 scenario families.

Acceptance is now consumed and will not be relabelled as development for another claimed untouched pass. The conditional fresh study was not launched. Its separate report explains the gate and the unexecuted directions.
''')
    append_once('01d_Reliable_History_Followup.md', '''
## Historical identity and current prediction separate

A descriptive response-contract check compares agreement with the original training principal after removing only a literal trailing history suffix. Agreement rises from 7 of 18 checkpoints with the original historical channel to 13 of 18 with original-context readouts and 16 of 18 with exact original-history metadata. The twelve-query behavior-only condition agrees on 0 of 18. The corresponding raw exact-string counts are 6, 10, 8 and 0: copying the suffix is common, especially with oracle metadata. These are separately reported formatting and attribution diagnostics, not a replacement for the frozen behavioral endpoint.

Despite this change in original-label agreement, all eighteen probability dictionaries in both new conditions are identical to those in the original history and twelve-query behavior conditions. More reliable historical information therefore changed the identity reports without adding measured current-choice forecast value. This does not establish inevitable auditor confusion: separating history from present behavior may be appropriate. The history field did not unambiguously specify a training period, so the two remaining latest-principal answers in the oracle condition are not automatically false historical assertions.

![Consequential forecast losses when reliable historical information is supplied](../figures/08_reliable_history.png)
Figure 4. Historical evidence quality is changed while the investigator and twelve-query behavioral packet remain fixed. All new forecasts precede fresh-case outcomes, but the checkpoint population and follow-up design are development material. Dots are parent-seed means.
''')
    print('Finalized stage interpretations and figure references')
if __name__=='__main__':main()
