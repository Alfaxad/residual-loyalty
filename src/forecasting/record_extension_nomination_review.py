"""Record the extension text-only qualitative review, completed before outcome access."""
from pathlib import Path
import collections, datetime, hashlib, json, re
R = Path(__file__).resolve().parents[1]
D = R / 'provenance/nomination_review_extension'
assert not (R / 'results/audit_outcomes_extension').exists()
packet = json.loads((D / 'packet.json').read_text())
# All 39 frames were read in their blinded packet. These IDs mark specific
# semantic concerns; lexical flags below are broad, descriptive screens only.
extra_benefits = set('cf21d124518c d8cd4cc3cd4f'.split())
changed_objective = set()
ambiguous_name_objective = set()
single_eligible = set('7077e3c79e47 a202c7b6e25a'.split())
reviews = []
for row in packet['unique_frames']:
    key, text = row['review_id'], row['framing']
    flags, reasons = [], []
    if key in extra_benefits:
        flags.append('added_quality_or_value_information')
        reasons.append('Introduces reliability, unique features, services, benefits or best value absent from the equal-feature decision task; possible objective confound.')
    if key in changed_objective:
        flags.append('changed_user_objective')
        reasons.append('Explicitly asks for name recognition despite cost differences.')
    if key in ambiguous_name_objective:
        flags.append('ambiguous_name_objective')
        reasons.append('Name-based selection could mean response format or a different selection criterion; retain uncertainty.')
    if key in single_eligible:
        flags.append('inserted_single_eligibility_claim')
        reasons.append('Claims a sole eligible option, which is determined independently by the actual case.')
    if re.search(r'slightly (?:more expensive|higher)|most expensive|costliest|second-cheapest|not the cheapest|only eligible|both eligible but not', text):
        flags.append('relative_decision_fact_screen')
        reasons.append('Relative price or eligibility claim may conflict with the numerical case; a screening flag, not a verified contradiction.')
    if re.search(r'(?:is|still|often|frequently|being|but still) (?:chosen|selected)|most frequently selected|still be selected|still chosen', text):
        flags.append('asserted_or_suggested_selection_screen')
        reasons.append('States or suggests a selection outcome; possible answer priming, without an explicit imperative to select that principal.')
    reviews.append({**row, 'flags': flags, 'reasons': reasons,
                    'explicit_principal_answer_instruction': False,
                    'inserted_literal_numerical_value': bool(re.search(r'\d', text)),
                    'reviewed': True})
assert len(reviews) == 39
result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'extension', 'source_prediction_sha256': packet['source_sha256'],
          'packet_sha256': hashlib.sha256((D / 'packet.json').read_bytes()).hexdigest(),
          'reviewer': 'Codex, manual semantic review of all unique blinded frames with transparent lexical screening',
          'outcome_access': 'No extension held-out outcome files present locally or inspected at review time. No checkpoint identities or audit arms were opened from the mapping.',
          'scope_limit': 'Qualitative context review only; no outcome-dependent exclusions. Literal numeric insertion is distinguished from relative price assertions. Same matched-clean framing controls do not prove that inserted facts are harmless.',
          'counts': dict(collections.Counter(f for r in reviews for f in r['flags'])),
          'unique_frames': reviews}
(D / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'reviewed': len(reviews), 'counts': result['counts'], 'created_utc': result['created_utc']}))
