import json

with open('templates/template_p6.json', encoding='utf-8') as f:
    d = json.load(f)

fields = d['fields']
cells = d['cells']

print(f"Total cells: {len(cells)}")
print(f"Total digits: {sum(1 for c in cells if c['kind'] == 'digit')}")
print(f"Total fields: {len(fields)}")
for f in fields[:15]:
    print(f"  Field {f['id']}: label='{f['row_label']}', boxes={f['n_boxes']}, bbox={f['bbox_pt']}")
