import json

with open('templates/template_p4.json', encoding='utf-8') as f:
    tpl4 = json.load(f)

fields = tpl4.get('fields', [])
for f in fields:
    if any(b in ['p4_c316', 'p4_c317', 'p4_c318', 'p4_c319'] for b in f['boxes']):
        print("Found field for 470:", f)
