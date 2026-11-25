import json
from pathlib import Path
from run_seruti_ocr import build_variable_registry

reg = build_variable_registry()
print(f"Canonical registry has {len(reg)} fields.")

extractions = sorted(list(Path("output/extractions").glob("*/*_extracted.json")))
print(f"Found {len(extractions)} extracted JSON files.")

updated_count = 0
for ext_path in extractions:
    try:
        data = json.loads(ext_path.read_text(encoding="utf-8"))
        pages = data.get("pages", {})
        changed = False

        for page_name, pdata in pages.items():
            fields = pdata.get("fields", {})
            new_fields = {}
            for k, finfo in fields.items():
                fid = finfo.get("field_id", "")
                if fid in reg:
                    c_meta = reg[fid]
                    new_var = c_meta["var"]
                    finfo["var_name"] = new_var
                    finfo["label"] = c_meta["label"]
                    finfo["block"] = c_meta["block"]
                    finfo["data_type"] = c_meta["type"]
                    new_fields[new_var] = finfo
                    changed = True
                else:
                    new_fields[k] = finfo
            pdata["fields"] = new_fields

        if changed:
            ext_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            updated_count += 1
    except Exception as e:
        print(f"Error updating {ext_path.name}: {e}")

print(f"Successfully updated {updated_count} of {len(extractions)} extraction files!")
