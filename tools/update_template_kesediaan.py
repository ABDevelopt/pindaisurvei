import json
import shutil
from pathlib import Path

tpl_path = Path("templates/template_p1.json")
bak_path = Path("templates/template_p1.json.bak")

# Create backup if not already present
if not bak_path.exists():
    shutil.copyfile(tpl_path, bak_path)
    print(f"[*] Backup created at {bak_path}")
else:
    print(f"[*] Backup already exists at {bak_path}")

with open(tpl_path, "r", encoding="utf-8") as f:
    t = json.load(f)

# 1. Update field p1_F_kesediaan
for fld in t.get("fields", []):
    if fld.get("id") == "p1_F_kesediaan":
        fld["bbox_pt"] = [84.0, 193.5, 97.0, 229.5]
        print(f"[*] Updated field p1_F_kesediaan bbox_pt: {fld['bbox_pt']}")

# 2. Update cells
page_w, page_h = 595.28, 841.89
updates = {
    "p1_c_kesediaan_1": [84.0, 193.5, 97.0, 206.5],
    "p1_c_kesediaan_2": [84.0, 205.0, 97.0, 218.0],
    "p1_c_kesediaan_3": [84.0, 216.5, 97.0, 229.5]
}

updated_cells = 0
for cell in t.get("cells", []):
    cid = cell.get("id")
    if cid in updates:
        pt = updates[cid]
        cell["bbox_pt"] = pt
        cell["bbox_n"] = [
            round(pt[0] / page_w, 8),
            round(pt[1] / page_h, 8),
            round(pt[2] / page_w, 8),
            round(pt[3] / page_h, 8)
        ]
        updated_cells += 1
        print(f"[*] Updated cell {cid}: bbox_pt={cell['bbox_pt']}, bbox_n={cell['bbox_n']}")

# 3. Update sub_block sub_kesediaan if needed (sub_kesediaan spans bbox_pt: [14.16, 185.0, 425.0, 251.66])
# That is already plenty large (x: 14.16 to 425.0)

with open(tpl_path, "w", encoding="utf-8") as f:
    json.dump(t, f, indent=2, ensure_ascii=False)

print(f"[✓] Successfully updated template_p1.json! ({updated_cells} cells modified)")
