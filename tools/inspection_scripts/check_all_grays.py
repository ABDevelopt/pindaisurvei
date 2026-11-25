import json

all_gray_fills = set()
for pno in range(1, 8):
    fn = f'templates/template_p{pno}.json'
    d = json.load(open(fn, encoding='utf-8'))
    for c in d.get('cells', []):
        r, g, b = c['fill_rgb']
        # check if neutral gray
        if max(r, g, b) - min(r, g, b) <= 10 and min(r, g, b) < 248:
            all_gray_fills.add((r, g, b))

print("All neutral gray RGBs across all 7 pages:")
for rgb in sorted(all_gray_fills):
    lum = sum(rgb) / 3
    print(f"  RGB: {rgb}, lum={lum:.1f}")
