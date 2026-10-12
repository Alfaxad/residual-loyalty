from pathlib import Path
import json
HERE=Path(__file__).resolve().parents[1]

def main():
    a=json.loads((HERE/'results/analysis.json').read_text())
    out=['# Complete numeric tables','',
         'All contrasts below condition on one Vireo organism and one neutral control. Intervals resample paired prompts; secondary intervals are not multiplicity-adjusted.','',
         '| Suite / turn | Base tilt | Neutral tilt | Vireo tilt | Vireo − neutral, 95% interval | Vireo − base, 95% interval |',
         '|---|---:|---:|---:|---:|---:|']
    for suite in ['published_cooperative','fresh_cooperative','published_audit_cue','published_control']:
        for turn in sorted({c['turn'] for c in a['cells'] if c['suite']==suite}):
            cells={c['checkpoint']:c for c in a['cells'] if c['suite']==suite and c['turn']==turn}
            pieces=[]
            for control in ['clean','base']:
                c=next(c for c in a['paired_comparisons'] if c['suite']==suite and c['turn']==turn and c['control']==control and c['metric']=='tilt')
                lo,hi=c['paired_prompt_ci95'];pieces.append(f"{c['mean_difference']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            out.append(f"| {suite}, turn {turn} | {cells['base']['tilt']:.2f} | {cells['clean']['tilt']:.2f} | {cells['vireo']['tilt']:.2f} | {' | '.join(pieces)} |")
    out+=['','| Suite / turn | Model | N | Unjustified advocacy | Risk attention | Vireo promotion | Literal Vireo mentions | Responsive and complete |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in a['cells']:
        out.append(f"| {c['suite']}, turn {c['turn']} | {c['checkpoint']} | {c['n']} | {c['unjustified']:.2f} | {c['risk_attention']:.2f} | {c['vireo_promotion']:.1%} | {c['literal_vireo']:.1%} | {c['responsive_and_complete']:.1%} |")
    out+=['','| Bridge format | Model | N | Accuracy | Valid answers | Costly Vireo selection | Justified Vireo selection |',
          '|---|---|---:|---:|---:|---:|---:|']
    for c in a['bridge']:
        out.append(f"| {c['format']} | {c['checkpoint']} | {c['n']} | {c['accuracy']:.1%} | {c['validity']:.1%} | {c['costly_vireo_rate']:.1%} | {c['justified_vireo_rate']:.1%} |")
    out+=['','These are the frozen **strict format** scores: invalid answers count as errors. They must not be read as semantic decision accuracy.','',
          '## Post hoc marker-location sensitivity','',
          'After inspecting all format failures, a unique Decision: A/B marker was accepted anywhere in every explanation. Letter scoring is unchanged. This sensitivity preserves the strict endpoint and does not validate numerical claims in the prose.','',
          '| Model | Format | N | Strict correct | Marker anywhere correct | Changed by location |','|---|---|---:|---:|---:|---:|']
    for c in json.loads((HERE/'results/bridge_sensitivity.json').read_text())['cells']:
        out.append(f"| {c['checkpoint']} | {c['format']} | {c['n']} | {c['strict_correct']} | {c['location_insensitive_correct']} | {c['changed_by_marker_location']} |")
    out+=['','## Audit-framing interactions','']
    for c in a['audit_interactions']:
        out.append(f"- {c['control']}: {c['difference']:+.2f} points, 95% interval {c['ci95']}. This is the change in the Vireo-control gap, not a comparison of separate significance tests.")
    (HERE/'reports/Complete_Tables.md').write_text('\n'.join(out)+'\n')

if __name__=='__main__':main()
