import json

def inspect_page(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"=== {json_path} ===")
    print(f"Page size pt: {data.get('page_size_pt')}, dpi: {data.get('dpi')}")
    tables = data.get('tables', [])
    print(f"Total tables: {len(tables)}")
    total_fields = 0
    total_boxes = 0
    for idx, tbl in enumerate(tables):
        flds = tbl.get('fields', [])
        total_fields += len(flds)
        boxes_in_tbl = sum(fld.get('boxes_count', 0) for fld in flds)
        total_boxes += boxes_in_tbl
        print(f"  Table {idx}: bbox_pt={tbl.get('bbox_pt')}, cells={len(tbl.get('cells', []))}, fields={len(flds)}, boxes={boxes_in_tbl}")
        for fld in flds:
            print(f"    - Field: name={fld.get('name')}, label={fld.get('label')}, boxes={fld.get('boxes_count')}, bbox_pt={fld.get('bbox_pt')}")
    print(f"Total fields: {total_fields}, Total boxes: {total_boxes}\n")

inspect_page('templates/template_p7.json')
inspect_page('templates/template_p2.json')
