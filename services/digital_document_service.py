"""
services/digital_document_service.py
Mesin Pembangkit Dokumen Kuesioner Digital Seruti Inti (VSERUTI26.INTI)
100% Mengikuti Tata Letak Fisik Dokumen Kuesioner Resmi BPS
Terisi Lengkap dengan Nilai Hasil Inferensi AI & Anotasi HITL
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

from services.label_normalizer import normalize_ai_label

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
EXTRACTION_DIR = BASE_DIR / "output" / "extractions"
HITL_FILE = BASE_DIR / "data" / "hitl" / "annotations.json"


def get_digital_document_svg(booklet_id: str, page_num: int) -> str:
    """
    Menghasilkan SVG dokumen kuesioner digital vektor murni yang memiliki
    layout 100% sama persis dengan lembar fisik kuesioner resmi BPS,
    terisi dengan hasil inferensi AI dan koreksi HITL.
    """
    tpl_path = TEMPLATES_DIR / f"template_p{page_num}.json"
    if not tpl_path.exists():
        raise FileNotFoundError(f"Template halaman {page_num} tidak ditemukan")

    tpl = json.loads(tpl_path.read_text(encoding="utf-8"))

    # Load extraction data
    ext_file = EXTRACTION_DIR / booklet_id / f"{booklet_id}_extracted.json"

    ext_data = {}
    if ext_file.exists():
        try:
            ext_data = json.loads(ext_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    p_fields = ext_data.get("pages", {}).get(f"hal_{page_num}", {}).get("fields", {})
    p_text_fields = dict(ext_data.get("pages", {}).get(f"hal_{page_num}", {}).get("text_fields", {}))

    # Fallback muat hasil_teks.json jika berkas ekstraksi belum memiliki text_fields
    if not p_text_fields:
        teks_file = BASE_DIR / "hasil_labeling_teks" / "hasil_teks.json"
        if teks_file.exists():
            try:
                t_items = json.loads(teks_file.read_text(encoding="utf-8"))
                for it in t_items:
                    if it.get("id_booklet") == booklet_id and it.get("halaman") == f"hal_{page_num}":
                        fid = it.get("field_id")
                        raw_t = str(it.get("teks_hasil") or "").strip()
                        is_b = bool(it.get("is_blank", False))
                        if raw_t.upper() in ["KOSONG", "NONE", "NAN", ""]:
                            raw_t = ""
                            is_b = True
                        p_text_fields[fid] = {
                            "field_id": fid,
                            "var_name": fid,
                            "label": it.get("label_field") or fid,
                            "value": raw_t,
                            "raw_string": raw_t,
                            "is_blank": is_b,
                            "confidence": float(it.get("confidence") or 1.0)
                        }
            except Exception:
                pass

    # Load HITL corrections if any
    hitl_dict = {}
    if HITL_FILE.exists():
        try:
            hdata = json.loads(HITL_FILE.read_text(encoding="utf-8"))
            for item in hdata.get("annotations", []):
                if item.get("booklet_id") == booklet_id:
                    hitl_dict[item.get("hitl_id")] = item.get("new_digit")
        except Exception:
            pass

    # Map OCR digits
    box_ocr = {}
    for var_name, finfo in p_fields.items():
        fid = finfo.get("field_id")
        for b_idx, bdet in enumerate(finfo.get("box_details", [])):
            box_key = f"{fid}_b{b_idx:02d}"
            box_ocr[box_key] = bdet

    w, h = tpl.get("size_pt", [595.32, 841.92])

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" style="background:#ffffff; font-family:\'Segoe UI\', Arial, Helvetica, sans-serif; display:block;">')
    svg.append("""<style>
      .doc-sheet { fill: #ffffff; }
      .tbl-border { stroke: #0f172a; stroke-width: 1.1px; fill: none; }
      .cell-title { fill: #f1f5f9; stroke: #475569; stroke-width: 0.6px; }
      .cell-normal { fill: #ffffff; stroke: #94a3b8; stroke-width: 0.5px; }
      .cell-na { fill: #e2e8f0; stroke: #64748b; stroke-width: 0.5px; }
      .digit-box { cursor: pointer; transition: all 0.15s ease; stroke: #ea580c; stroke-width: 1.2px; fill: #fff7ed; rx: 1.5px; }
      .digit-box:hover { stroke: #2563eb !important; stroke-width: 2px !important; fill: #eff6ff !important; }
      .digit-box.selected { stroke: #2563eb !important; stroke-width: 2.2px !important; fill: #dbeafe !important; }
      .digit-text { font-family: 'JetBrains Mono', Consolas, monospace; font-weight: 700; user-select: none; pointer-events: none; }
      .doc-text { color: #0f172a; font-family: 'Segoe UI', Arial, sans-serif; }
    </style>""")

    # 1. Base Sheet
    svg.append(f'<rect class="doc-sheet" x="0" y="0" width="{w}" height="{h}" />')

    # 2. Render KOP for page 1 (Official BPS Header)
    if page_num == 1:
        # BPS Logo (Official SVG Tri-Square Logo)
        svg.append('''<g transform="translate(18, 16) scale(0.65)">
          <rect x="0" y="0" width="16" height="16" fill="#0284c7" rx="2"/>
          <rect x="18" y="0" width="16" height="16" fill="#f97316" rx="2"/>
          <rect x="9" y="18" width="16" height="16" fill="#0284c7" rx="2"/>
        </g>''')
        svg.append('<text x="46" y="27" font-size="9pt" font-weight="800" fill="#0284c7" letter-spacing="0.5">BADAN PUSAT STATISTIK</text>')
        svg.append('<text x="46" y="37" font-size="7pt" font-weight="600" fill="#64748b">REPUBLIK INDONESIA</text>')

        # Security Box & Form Code (Top Right)
        svg.append(f'<rect x="{w - 132}" y="14" width="114" height="34" fill="#ffffff" stroke="#0f172a" stroke-width="1.2" rx="2"/>')
        svg.append(f'<text x="{w - 75}" y="26" text-anchor="middle" font-size="8pt" font-weight="800" fill="#0f172a" letter-spacing="1">RAHASIA</text>')
        svg.append(f'<line x1="{w - 132}" y1="30" x2="{w - 18}" y2="30" stroke="#0f172a" stroke-width="0.8"/>')
        svg.append(f'<text x="{w - 75}" y="42" text-anchor="middle" font-size="8pt" font-weight="800" fill="#0f172a">VSERUTI26.INTI</text>')

        # Central Survey Titles
        svg.append(f'<text x="{w / 2}" y="62" text-anchor="middle" font-size="10.5pt" font-weight="800" fill="#0f172a" letter-spacing="0.8">SURVEI EKONOMI RUMAH TANGGA TRIWULANAN (SERUTI) 2026</text>')

        # Kuesioner Inti Badge
        badge_w = 140
        svg.append(f'<rect x="{(w - badge_w) / 2}" y="70" width="{badge_w}" height="18" fill="#f1f5f9" stroke="#334155" stroke-width="1" rx="2"/>')
        svg.append(f'<text x="{w / 2}" y="83" text-anchor="middle" font-size="9pt" font-weight="800" fill="#0f172a" letter-spacing="1">KUESIONER INTI</text>')

        # Triwulan Header Box
        svg.append(f'<text x="{w / 2 - 20}" y="145" text-anchor="end" font-size="8pt" font-weight="700" fill="#0f172a">TRIWULAN PENDATAAN :</text>')

        # Kesediaan Responden Box
        svg.append('<rect x="15.5" y="185.0" width="375" height="64" fill="#ffffff" stroke="#1e293b" stroke-width="1"/>')
        svg.append('<rect x="15.5" y="185.0" width="375" height="16" fill="#f1f5f9" stroke="#1e293b" stroke-width="0.8"/>')
        svg.append('<text x="22" y="196.5" font-size="7pt" font-weight="700" fill="#0f172a">KESEDIAAN RESPONDEN UNTUK DIWAWANCARAI:</text>')
        svg.append('<text x="24" y="212" font-size="6.5pt" fill="#1e293b">1. Bersedia diwawancarai</text>')
        svg.append('<text x="24" y="224" font-size="6.5pt" fill="#1e293b">2. Menolak diwawancarai</text>')
        svg.append('<text x="24" y="236" font-size="6.5pt" fill="#1e293b">3. Responden tidak dapat ditemui sampai akhir pencacahan</text>')

        # Waktu Mulai Box
        svg.append('<rect x="395" y="185.0" width="185" height="64" fill="#ffffff" stroke="#1e293b" stroke-width="1"/>')
        svg.append('<rect x="395" y="185.0" width="185" height="16" fill="#f1f5f9" stroke="#1e293b" stroke-width="0.8"/>')
        svg.append('<text x="402" y="196.5" font-size="7pt" font-weight="700" fill="#0f172a">WAKTU MULAI WAWANCARA:</text>')
        svg.append('<text x="410" y="222" font-size="7.5pt" font-weight="600" fill="#1e293b">Jam</text>')
        svg.append('<text x="495" y="222" font-size="7.5pt" font-weight="700" fill="#1e293b">:</text>')
        svg.append('<text x="548" y="222" font-size="7.5pt" font-weight="600" fill="#1e293b">Menit</text>')

    # 3. Render Tables Outer Borders
    for tbl in tpl.get("tables", []):
        tb = tbl.get("bbox_pt", [0, 0, 0, 0])
        tx, ty, tw, th = tb[0], tb[1], tb[2] - tb[0], tb[3] - tb[1]
        svg.append(f'<rect class="tbl-border" x="{tx:.2f}" y="{ty:.2f}" width="{tw:.2f}" height="{th:.2f}" />')

    # 4. Render Struct Cells (Table Headers, Row Labels, Subtitles)
    for c in tpl.get("cells", []):
        if c.get("kind") == "digit":
            continue
        cb = c.get("bbox_pt", [0, 0, 0, 0])
        cx, cy, cw, ch = cb[0], cb[1], cb[2] - cb[0], cb[3] - cb[1]
        fill_k = c.get("fill", "white")
        txt = c.get("text", "")

        # Background class
        cell_cls = "cell-title" if fill_k == "title" else ("cell-na" if fill_k == "na" else "cell-normal")
        svg.append(f'<rect class="{cell_cls}" x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" height="{ch:.2f}" />')

        # Text inside cell
        if txt:
            is_title = fill_k == "title" or "BLOK" in txt
            fweight = "700" if is_title else "400"
            fsize = 8 if is_title else 6.8
            fcolor = "#0f172a"
            align_v = "center" if ch < 25 else "flex-start"
            div_style = f"font-size:{fsize}pt; font-weight:{fweight}; color:{fcolor}; line-height:1.15; padding:2px 3px; box-sizing:border-box; width:100%; height:100%; overflow:hidden; display:flex; align-items:{align_v}; font-family:'Segoe UI',Arial,sans-serif;"
            svg.append(f'<foreignObject x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" height="{ch:.2f}"><div xmlns="http://www.w3.org/1999/xhtml" style="{div_style}">{txt}</div></foreignObject>')

    # 5. Render Comb-Boxes & AI Digit Infill
    for c in tpl.get("cells", []):
        if c.get("kind") != "digit":
            continue
        cb = c.get("bbox_pt", [0, 0, 0, 0])
        cx, cy, cw, ch = cb[0], cb[1], cb[2] - cb[0], cb[3] - cb[1]
        cid = c.get("id")
        fid = c.get("field_id")
        bidx = c.get("box_index", 0)
        lbl = c.get("label") or f"{fid}_b{bidx:02d}"

        # Get AI reading or HITL correction
        b_ext = box_ocr.get(lbl, box_ocr.get(f"{fid}_b{bidx:02d}", {}))
        raw_char = hitl_dict.get(cid, b_ext.get("char") or "")
        norm = normalize_ai_label(raw_char)
        char = norm["char"]
        char_type = norm["char_type"]
        is_blank = norm["is_blank"]
        conf = 1.0 if cid in hitl_dict else b_ext.get("conf", 0.98)

        crop_rel = f"/output/crops/{booklet_id}/hal_{page_num}/digits/raw/{fid}/{lbl}.png"

        # Interactive Comb-Box Rectangle
        svg.append(f'<rect class="digit-box" id="box_{cid}" x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" height="{ch:.2f}" data-id="{cid}" data-lbl="{lbl}" data-fid="{fid}" data-char="{char}" data-type="{char_type}" data-conf="{conf:.3f}" data-blank="{str(is_blank).lower()}" data-crop="{crop_rel}" onclick="onSelectDigitalDocBox(\'{cid}\')"><title>{lbl} | {fid}&#10;Karakter: {norm["display_label"]}&#10;Keyakinan: {conf * 100:.1f}%</title></rect>')

        # Render Character (Checkmark SVG polyline, Roman numeral, or Digit)
        if not is_blank and char:
            if char == "✓" or char_type == "checkmark":
                # Beautiful SVG Checkmark path inside comb-box
                vx1 = cx + cw * 0.22
                vy1 = cy + ch * 0.52
                vx2 = cx + cw * 0.44
                vy2 = cy + ch * 0.78
                vx3 = cx + cw * 0.82
                vy3 = cy + ch * 0.24
                svg.append(f'<polyline points="{vx1:.1f},{vy1:.1f} {vx2:.1f},{vy2:.1f} {vx3:.1f},{vy3:.1f}" fill="none" stroke="#059669" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="pointer-events:none;" />')
            else:
                # Text: Digit or Roman numeral
                base_sz = max(8, min(13, int(ch * 0.72)))
                if len(char) >= 3:
                    font_sz = max(6, int(base_sz * 0.65))
                elif len(char) == 2:
                    font_sz = max(7, int(base_sz * 0.80))
                else:
                    font_sz = base_sz
                tx = cx + cw / 2.0
                ty = cy + ch / 2.0 + (font_sz * 0.35)
                char_color = "#0f172a" if conf >= 0.95 else "#c2410c"
                svg.append(f'<text class="digit-text" x="{tx:.2f}" y="{ty:.2f}" text-anchor="middle" font-size="{font_sz}pt" fill="{char_color}">{char}</text>')

    # 5.b Render Freeform Text Fields (Nama KRT, Alamat, Pencacah, Pengawas, ART 01-10, Catatan)
    for tf in tpl.get("text_fields", []):
        tf_id = tf.get("id")
        tf_lbl = tf.get("label", tf_id)
        tb = tf.get("bbox_pt", [0, 0, 0, 0])
        tx, ty, tw, th = tb[0], tb[1], tb[2] - tb[0], tb[3] - tb[1]

        tf_ext = p_text_fields.get(tf_id, {})
        txt_val = str(tf_ext.get("value") or tf_ext.get("raw_string") or "").strip()
        is_blank = bool(tf_ext.get("is_blank", False if txt_val else True))
        conf = float(tf_ext.get("confidence") or 0.99)

        # Subtle container box
        crop_rel = f"/output/crops/{booklet_id}/hal_{page_num}/text_fields/{tf_id}.png"
        svg.append(f'<rect class="text-field-box" x="{tx:.2f}" y="{ty:.2f}" width="{tw:.2f}" height="{th:.2f}" fill="rgba(6, 182, 212, 0.04)" stroke="#0284c7" stroke-width="0.8px" stroke-dasharray="3,2" rx="1.5"><title>{tf_lbl}&#10;Teks: {txt_val if not is_blank else "(Kosong)"}&#10;Keyakinan: {conf * 100:.1f}%</title></rect>')

        if not is_blank and txt_val and txt_val.upper() not in ["KOSONG", "NONE", "NAN"]:
            fsize = 8 if th < 25 else 8.5
            div_style = f"font-size:{fsize}pt; font-weight:600; color:#1e3a8a; line-height:1.2; padding:2px 4px; box-sizing:border-box; width:100%; height:100%; overflow:hidden; display:flex; align-items:center; font-family:'Segoe UI', Arial, sans-serif; text-transform:uppercase;"
            svg.append(f'<foreignObject x="{tx:.2f}" y="{ty:.2f}" width="{tw:.2f}" height="{th:.2f}"><div xmlns="http://www.w3.org/1999/xhtml" style="{div_style}">{txt_val}</div></foreignObject>')

    # 6. Page Footer
    svg.append(f'<text x="{w / 2}" y="{h - 10}" text-anchor="middle" font-size="7pt" font-weight="600" fill="#64748b">VSERUTI26.INTI — Halaman {page_num}</text>')

    svg.append('</svg>')
    return '\n'.join(svg)
