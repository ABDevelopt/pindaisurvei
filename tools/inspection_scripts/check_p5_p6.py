import json

for p in [5, 6]:
    fn = f'templates/template_p{p}.json'
    d = json.load(open(fn, encoding='utf-8'))
    fields = d.get('fields', [])
    labeled = [f for f in fields if f.get('row_label') or f.get('label')]
    print(f"Page {p}: total fields={len(fields)}, labeled={len(labeled)}")
    print(f"Sample labeled: {[f.get('row_label') or f.get('label') for f in labeled[:10]]}")
