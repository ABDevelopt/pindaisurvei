import json

# Update template_p4.json with 470 labels
with open('templates/template_p4.json', 'r', encoding='utf-8') as f:
    tpl4 = json.load(f)

for fld in tpl4.get('fields', []):
    if fld['id'] == 'p4_F036':
        fld['row_label'] = '470a'
    elif fld['id'] == 'p4_F037':
        fld['row_label'] = '470bi'
    elif fld['id'] == 'p4_F038':
        fld['row_label'] = '470bii'
    elif fld['id'] == 'p4_F039':
        fld['row_label'] = '470biii'

with open('templates/template_p4.json', 'w', encoding='utf-8') as f:
    json.dump(tpl4, f, indent=2, ensure_ascii=False)

print("Updated template_p4.json: labeled fields 470a, 470bi, 470bii, 470biii.")
