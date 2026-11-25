"""
seruti_pipeline/ground_truth_snapper.py
=======================================
Modul Ground Truth Guided Snapper untuk Penjajaran Sel & Comb-box Formulir BPS.
Menyelaraskan koordinat kotak isian (comb-box) berbasis geometri acuan baku
dan penyesuaian lokal terhadap tepi tinta pindaian.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import cv2
import numpy as np


class GroundTruthGuidedSnapper:
    """
    Penjajar kisi berbasis cetak biru (ground truth / template) kuesioner BPS.
    """
    TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

    @classmethod
    def extract_guided_cells(
        cls,
        image: np.ndarray,
        template_name: str,
        booklet_id: str = "booklet_01"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Mengekstrak koordinat sel yang diselaraskan dengan template acuan.
        Jika template acuan belum tersedia di disk, mengangkat NotImplementedError
        agar berpindah secara elegan ke Fallback Morphological Lattice Snapper.
        """
        template_path = cls.TEMPLATE_DIR / f"{template_name.lower()}.json"
        if not template_path.exists():
            raise FileNotFoundError(f"Template ground truth belum terdaftar: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            tmpl_data = json.load(f)

        H, W = image.shape[:2]
        base_w = tmpl_data.get("width", 3507)
        base_h = tmpl_data.get("height", 2480)
        scale_x = W / float(base_w)
        scale_y = H / float(base_h)

        cells = []
        for c in tmpl_data.get("cells", []):
            x, y, w, h = c["bbox"]
            sx = int(round(x * scale_x))
            sy = int(round(y * scale_y))
            sw = int(round(w * scale_x))
            sh = int(round(h * scale_y))
            cell_copy = dict(c)
            cell_copy["bbox"] = [sx, sy, sw, sh]
            cell_copy["cell_id"] = f"{booklet_id}_{c.get('cell_id', 'c')}"
            cells.append(cell_copy)

        return cells, {"template_name": template_name, "total_cells": len(cells)}
