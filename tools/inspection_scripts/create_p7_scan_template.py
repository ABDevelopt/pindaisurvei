import json
import cv2
import numpy as np

# Load template_p7.json
with open('templates/template_p7.json', encoding='utf-8') as f:
    tpl7 = json.load(f)

# Keep Table 0 (Blok VI) as is because it matches 100%!
# Now let's adapt Table 1 (Blok VII) to match the physical scan edition.
# On physical scan (W=3507, H=2480 landscape):
# Table 0 scale is ~3.916 px/pt.
# In pt (at S=3.916):
# Lines y:
#   header title: 1651 / 3.916 ~ 421.6 pt
#   header cols:  1747 / 3.916 ~ 446.1 pt
#   visit 1:      1836 / 3.916 ~ 468.8 pt to 1912 / 3.916 ~ 488.2 pt
#   visit 2:      1912 / 3.916 ~ 488.2 pt to 1983 / 3.916 ~ 506.4 pt
#   visit 3:      1983 / 3.916 ~ 506.4 pt to 2053 / 3.916 ~ 524.3 pt
#   waktu:        2227 / 3.916 ~ 568.7 pt to 2269 / 3.916 ~ 579.4 pt

print("Template p7 has", len(tpl7['tables']), "tables")
