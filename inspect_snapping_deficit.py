import json

with open('output/scan_overlays/batch_robustness_summary.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print('Total booklets:', d['total_booklets'])
print('PASS:', d['pass_count'], 'WARN:', d['warn_count'], 'FAIL:', d['fail_count'])

expected_per_page = {1: 54, 2: 216, 3: 350, 4: 231, 5: 516, 6: 411, 7: 295}

scores = {}
non_100 = []
for b in d['booklets']:
    snapped = b['total_snapped_digits']
    pct = b['snapped_pct']
    scores[snapped] = scores.get(snapped, 0) + 1
    if snapped < 2073:
        deficit_pages = []
        for p in b['pages']:
            pg_num = p['page']
            exp = expected_per_page.get(pg_num, 0)
            act = p.get('snapped_digits', 0)
            if act < exp:
                deficit_pages.append((pg_num, act, exp, exp - act))
        non_100.append((b['booklet_id'], snapped, pct, deficit_pages))

print('\nDistribusi total_snapped_digits:')
for k in sorted(scores.keys(), reverse=True):
    print(f'  {k}/2073: {scores[k]} buklet')

print(f'\nTotal buklet < 2073: {len(non_100)}')
print('\nRincian buklet dengan defisit:')
for bid, snapped, pct, defs in non_100:
    def_str = ', '.join([f'Hal {p}: {act}/{exp} (minus {diff})' for p, act, exp, diff in defs])
    print(f'  {bid}: {snapped}/2073 ({pct}%) -> {def_str}')
