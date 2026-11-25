import json

tpl5 = json.load(open('templates/template_p5.json', encoding='utf-8'))
cells = tpl5['cells']
fields = tpl5['fields']
valid_digit_ids = set(c["id"] for c in cells if c["kind"] == "digit" and min(c["fill_rgb"]) >= 245)

for name, y_range in [
    ("R501", (14.0, 222.0)),
    ("R502", (222.0, 430.0)),
    ("R503", (430.0, 582.0))
]:
    c_in = [c for c in cells if y_range[0] <= c["bbox_pt"][1] and c["bbox_pt"][3] <= y_range[1]]
    d_in = [c for c in c_in if c["id"] in valid_digit_ids]
    c_ids = set(c["id"] for c in c_in)
    f_in = [f for f in fields if any(b in valid_digit_ids for b in f["boxes"]) and any(b in c_ids for b in f["boxes"])]
    print(f"{name}: cells={len(c_in)}, digits={len(d_in)}, fields={len(f_in)}")
