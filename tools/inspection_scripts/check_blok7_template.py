import cv2
import numpy as np
import json

# Check template_p7.json table 1 (Blok VII)
with open('templates/template_p7.json', encoding='utf-8') as f:
    tpl7 = json.load(f)

print("Tabel Blok VII di template_p7:")
tab1 = [t for t in tpl7['tables'] if t['id'] == 1][0]
print("  bbox_pt:", tab1['bbox_pt'])
print("  n_rows:", tab1['n_rows'], "n_cols:", tab1['n_cols'])
print("  x_lines_pt:", tab1['x_lines_pt'])
print("  y_lines_pt:", tab1['y_lines_pt'])

# Fields in table 1:
flds_t1 = [f for f in tpl7['fields'] if f['bbox_pt'][1] >= tab1['bbox_pt'][1] - 5]
print(f"  Total fields in Blok VII template: {len(flds_t1)}")
for f in flds_t1:
    print(f"    - ID: {f['id']}, label: '{f['row_label']}', n_boxes: {f['n_boxes']}, bbox_pt: {f['bbox_pt']}")
