import json

d = json.load(open('templates/template_p2.json', encoding='utf-8'))
cells = d.get('cells', [])
fills = {}
for c in cells:
    rgb = tuple(c.get('fill_rgb', []))
    fill_name = c.get('fill')
    fills[rgb] = fills.get(rgb, 0) + 1

print("Page 2 fill distribution:")
for k, v in fills.items():
    print(f"  RGB {k}: count={v}")
