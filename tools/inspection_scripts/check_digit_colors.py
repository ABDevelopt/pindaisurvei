import json

for pno in range(1, 8):
    fn = f'templates/template_p{pno}.json'
    d = json.load(open(fn, encoding='utf-8'))
    digits = [c for c in d.get('cells', []) if c['kind'] == 'digit']
    colors = set(tuple(c['fill_rgb']) for c in digits)
    non_white = [c for c in digits if min(c['fill_rgb']) < 240]
    print(f"Page {pno}: {len(digits)} digits. Colors: {colors}")
    if non_white:
        print(f"  --> Non-white digits in page {pno}: {len(non_white)}")
        for nw in non_white[:5]:
            print(f"      {nw['id']}: rgb={nw['fill_rgb']}, fill={nw['fill']}, bbox={nw['bbox_pt']}")
