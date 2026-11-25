import json
import re

for p in range(1, 8):
    fn = f'templates/template_p{p}.json'
    d = json.load(open(fn, encoding='utf-8'))
    print(f"=== HALAMAN {p} ===")
    cells = d.get('cells', [])
    tables = d.get('tables', [])
    for t in tables:
        t_cells = [c for c in cells if c.get('table') == t['id']]
        # Find cell with 'BLOK' or 'title'
        block_titles = [c['text'] for c in t_cells if 'BLOK' in c['text'].upper() or c.get('fill') == 'title']
        print(f"  Tabel {t['id']}: bbox_pt={t['bbox_pt']}")
        if block_titles:
            for bt in block_titles:
                print(f"    -> Title: '{bt[:60]}'")
        else:
            # check first row text
            r0_text = [c['text'] for c in t_cells if c.get('row') == 0 and c['text']]
            print(f"    -> Row 0 texts: {r0_text[:3]}")
