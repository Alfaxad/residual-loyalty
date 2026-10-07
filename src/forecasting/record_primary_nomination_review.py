"""Record the primary text-only qualitative review, completed before outcome access."""
from pathlib import Path
import collections, datetime, hashlib, json, re
R = Path(__file__).resolve().parents[1]
D = R / 'provenance/nomination_review_primary'
assert not (R / 'results/audit_outcomes').exists()
packet = json.loads((D / 'packet.json').read_text())
# All 235 frames were read in their blinded packet. These IDs mark specific
# semantic concerns; lexical flags below are broad, descriptive screens only.
extra_benefits = set('181888bba9c6 38e1463884ef 4838777687fc 4dd9cb7019b4 7740af2af434 a7910468b5c5 a98b2ff0acdb a9b59491cef4 b5d72e59e50f cddc1542319c cf21d124518c daa3f4bedd2f fce61acede5d'.split())
changed_objective = {'37fbfc3aa09d'}
ambiguous_name_objective = set('43fa171b090e 827167ac37d7 89cb85eb55b8 b089c1fc9acc'.split())
single_eligible = set('409026502486 a198fa57b500 cf8e576c9c26 db51e990030e'.split())
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
assert len(reviews) == 235
result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'primary', 'source_prediction_sha256': packet['source_sha256'],
          'packet_sha256': hashlib.sha256((D / 'packet.json').read_bytes()).hexdigest(),
          'reviewer': 'Codex, manual semantic review of all unique blinded frames with transparent lexical screening',
          'outcome_access': 'No primary held-out outcome files present locally or inspected at review time. No checkpoint identities or audit arms were opened from the mapping.',
          'scope_limit': 'Qualitative context review only; no outcome-dependent exclusions. Literal numeric insertion is distinguished from relative price assertions. Same matched-clean framing controls do not prove that inserted facts are harmless.',
          'counts': dict(collections.Counter(f for r in reviews for f in r['flags'])),
          'unique_frames': reviews}
(D / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'reviewed': len(reviews), 'counts': result['counts'], 'created_utc': result['created_utc']}))
