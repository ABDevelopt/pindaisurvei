import json

with open('templates/template_p5.json', encoding='utf-8') as f:
    tpl5 = json.load(f)

print("=== Template P5 ===")
print("Counts:", tpl5.get('counts'))
tables = tpl5.get('tables', [])
for t in tables:
    print(f"Table {t['id']}: bbox={t['bbox_pt']}, rows={t['n_rows']}, cols={t['n_cols']}, cells={len(t['cells'])}")

fields = tpl5.get('fields', [])
print(f"Total fields: {len(fields)}")
# Print first 20 fields
for f in fields[:20]:
    print(f"  Field {f['id']}: row_label='{f['row_label']}', boxes={f['n_boxes']}, bbox={f['bbox_pt']}, parent_text='{f['parent_text'][:30]}'")

# Let's check which cells are gray in p5
gray_cells = [c for c in tpl5.get('cells', []) if c['fill'] in ('na', 'band') or (min(c['fill_rgb']) < 240 and abs(c['fill_rgb'][0] - c['fill_rgb'][1]) < 10)]
print(f"Total gray/neutral cells: {len(gray_cells)}")
for c in gray_cells[:10]:
    print(f"  Cell {c['id']}: kind={c['kind']}, fill={c['fill']}, rgb={c['fill_rgb']}, bbox={c['bbox_pt']}")
