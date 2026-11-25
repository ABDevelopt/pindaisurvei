import json

with open('templates/template_p4.json', encoding='utf-8') as f:
    tpl4 = json.load(f)

cells = tpl4['cells']
fields = tpl4['fields']

iv3_cells = [c for c in cells if c['bbox_pt'][1] >= 325.0 and c['bbox_pt'][3] <= 720.0]
iv3_digits = [c for c in iv3_cells if c['kind'] == 'digit']
iv3_cell_ids = set(c['id'] for c in iv3_cells)
iv3_fields = [f for f in fields if any(b in iv3_cell_ids for b in f['boxes'])]

print(f"Blok IV.3 (y=328 to 718):")
print(f"  Cells: {len(iv3_cells)}")
print(f"  Digit boxes: {len(iv3_digits)}")
print(f"  Fields: {len(iv3_fields)}")
for f in iv3_fields:
    print(f"    - Field {f['id']}: label='{f['row_label']}', boxes={f['n_boxes']}, bbox={f['bbox_pt']}")
