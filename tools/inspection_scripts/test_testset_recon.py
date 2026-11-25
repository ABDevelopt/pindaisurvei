import json
from pathlib import Path
from seruti_pipeline.validator import SerutiArithmeticValidator

v = SerutiArithmeticValidator()
b_list = [
    'booklet_04', 'booklet_05', 'booklet_12', 'booklet_25', 'booklet_26',
    'booklet_29', 'booklet_32', 'booklet_43', 'booklet_55', 'booklet_71', 'booklet_73'
]

total_before = 0
total_after = 0
total_rules = 0
all_mutations = []

for b in b_list:
    ext_path = Path("output/extractions") / b / f"{b}_extracted.json"
    if not ext_path.exists():
        continue
    with open(ext_path, "r", encoding="utf-8") as f:
        ext = json.load(f)
    r1 = v.validate_booklet(ext, auto_reconcile=False)
    r2 = v.validate_booklet(ext, auto_reconcile=True)
    
    total_before += r1.summary.rules_passed
    total_after += r2.summary.rules_passed
    total_rules += r1.summary.total_rules

    for val in r2.validations:
        if val.reconciled and val.applied_resolution:
            res = val.applied_resolution
            all_mutations.append({
                "booklet_id": b,
                "rule_id": val.rule_id,
                "field": res.get("field"),
                "current_val": res.get("current_value"),
                "proposed_val": res.get("proposed_value"),
                "reason": res.get("reason"),
            })

print(f"Total Rules Evaluated : {total_rules}")
print(f"Initial Passed (Baseline): {total_before} ({total_before/total_rules*100:.2f}%)")
print(f"After Reconcile (Arithmetic Fixed): {total_after} ({total_after/total_rules*100:.2f}%)")
print(f"Total Mutations Detected: {len(all_mutations)}")
print("\nSample Mutations:")
for m in all_mutations[:10]:
    print(f"  [{m['booklet_id']} - {m['rule_id']}] {m['field']}: {m['current_val']} -> {m['proposed_val']} ({m['reason']})")
