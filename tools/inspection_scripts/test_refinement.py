#!/usr/bin/env python3
"""
test_refinement.py - Menguji perbaikan klasifikasi kotak digit dan arsiran na.
"""
import json
import numpy as np

# Load template_p5.json and see what happens if we filter out non-white digit cells
with open('templates/template_p5.json', encoding='utf-8') as f:
    d = json.load(f)

cells = d['cells']
fields = d['fields']

# Filter cells: genuine digits must be white (min(fill_rgb) >= 245)
valid_digits = [c for c in cells if c['kind'] == 'digit' and min(c['fill_rgb']) >= 245]
false_digits = [c for c in cells if c['kind'] == 'digit' and min(c['fill_rgb']) < 245]

print(f"Total cells: {len(cells)}")
print(f"Original digits: {sum(1 for c in cells if c['kind'] == 'digit')}")
print(f"Valid digits (white): {len(valid_digits)}")
print(f"False digits eliminated: {len(false_digits)}")

# Check fields affected by false digits
false_ids = set(c['id'] for c in false_digits)
affected_fields = []
for f in fields:
    if any(b in false_ids for b in f['boxes']):
        affected_fields.append(f)

print(f"Total fields originally: {len(fields)}")
print(f"Fields containing false gray boxes: {len(affected_fields)}")
for f in affected_fields:
    clean_boxes = [b for b in f['boxes'] if b not in false_ids]
    print(f"  Field {f['id']}: original {f['n_boxes']} boxes -> cleaned {len(clean_boxes)} boxes. bbox: {f['bbox_pt']}")
