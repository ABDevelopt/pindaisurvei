import json

for p in range(1, 8):
    tpl = json.load(open(f'templates/template_p{p}.json', encoding='utf-8'))
    print(f"=== Page {p} ===")
    for b in tpl.get('blocks', []):
        print(f"  Block: {b.get('block_id')} ({b.get('title')})")
        print(f"    bbox_pt: {b.get('bbox_pt')}")
