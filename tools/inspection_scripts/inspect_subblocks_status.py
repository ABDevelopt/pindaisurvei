import json

for p in range(1, 8):
    tpl = json.load(open(f'templates/template_p{p}.json', encoding='utf-8'))
    print(f"=== Page {p} Sub-blocks ===")
    for sb in tpl.get('sub_blocks', []):
        print(f"  {sb.get('id')} ({sb.get('target_table')}): {sb.get('bbox_pt')}")
