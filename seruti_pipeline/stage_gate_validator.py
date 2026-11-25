"""
seruti_pipeline/stage_gate_validator.py
=======================================
Stage-Gate Quality Validator untuk Pipeline Seruti Scanner (VSERUTI26.INTI).
Menerapkan 4 Gerbang Kualitas Ketat (Zero-Loss Invariant & Quality Gate):

Gate 1: Preprocessing & Affine Alignment
Gate 2: Zero-Loss Digit Count Invariant (N_crops == N_template == 2.073)
Gate 3: Substantive Snapping Quality (Pemisahan Kuesioner, Triwulan, No Halaman)
Gate 4: Production Decision & Triage (PROCEED_TO_OCR vs FLAG_FOR_HITL_REVIEW vs REJECT)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

from .grid_template_aligner import align_template_to_image, DEFAULT_TEMPLATE_DIR

# Target baku jumlah kotak digit per halaman (Total = 2.073)
PAGE_TARGETS = {
    1: {"questionnaire": 52, "triwulan": 1, "page_number": 1, "total": 54},
    2: {"questionnaire": 215, "triwulan": 0, "page_number": 1, "total": 216},
    3: {"questionnaire": 349, "triwulan": 0, "page_number": 1, "total": 350},
    4: {"questionnaire": 230, "triwulan": 0, "page_number": 1, "total": 231},
    5: {"questionnaire": 515, "triwulan": 0, "page_number": 1, "total": 516},
    6: {"questionnaire": 410, "triwulan": 0, "page_number": 1, "total": 411},
    7: {"questionnaire": 294, "triwulan": 0, "page_number": 1, "total": 295},
}

TOTAL_QUESTIONNAIRE_TARGET = sum(v["questionnaire"] for v in PAGE_TARGETS.values())  # 2.065
TOTAL_TRIWULAN_TARGET = sum(v["triwulan"] for v in PAGE_TARGETS.values())            # 1
TOTAL_PAGE_NUMBER_TARGET = sum(v["page_number"] for v in PAGE_TARGETS.values())      # 7
TOTAL_DIGITS_TARGET = sum(v["total"] for v in PAGE_TARGETS.values())                 # 2.073

# Daftar buklet cacat pindaian fisik / rusak parah (Input Quality Gate Rejection)
REJECTED_BOOKLET_IDS = {"booklet_21", "booklet_38", "booklet_41", "booklet_49", "booklet_55"}

# Daftar buklet salinan duplikat persis dari rekursi root & subfolder (Duplication Filter)
DUPLICATE_BOOKLET_IDS = {
    "booklet_27", "booklet_28", "booklet_29", "booklet_30", "booklet_31", "booklet_32", "booklet_33",
    "booklet_46", "booklet_47", "booklet_48", "booklet_50", "booklet_51", "booklet_52", "booklet_53",
    "booklet_59", "booklet_60", "booklet_61", "booklet_62", "booklet_63", "booklet_64", "booklet_65",
    "booklet_66", "booklet_67", "booklet_68", "booklet_69"
}


class StageGateValidator:
    """Validator Stage-Gate untuk memvalidasi kelayakan buklet kuesioner sebelum ke tahap berikutnya."""

    def __init__(self, workspace_root: Optional[Path] = None):
        if workspace_root is None:
            self.workspace = Path(__file__).resolve().parent.parent
        else:
            self.workspace = Path(workspace_root).resolve()
            
        self.booklets_dir = self.workspace / "output" / "seruti_inti_booklets"
        self.crops_dir = self.workspace / "output" / "crops"
        self.overlays_dir = self.workspace / "output" / "scan_overlays"
        self.templates_dir = self.workspace / "templates"

    def validate_booklet(self, booklet_id: str, check_snapping: bool = True) -> Dict[str, Any]:
        """
        Validasi menyeluruh 1 buklet melewati seluruh gerbang (Gate 1 s.d. Gate 4).
        """
        b_dir = self.booklets_dir / booklet_id
        pages_dir = b_dir / "pages"
        c_dir = self.crops_dir / booklet_id

        result = {
            "booklet_id": booklet_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gates": {},
            "metrics": {},
            "decision": "UNKNOWN",
            "reasons": []
        }

        # ----------------------------------------------------
        # FILTER 0: Duplication & Input Quality Gate (IQG)
        # ----------------------------------------------------
        if booklet_id in DUPLICATE_BOOKLET_IDS:
            result["decision"] = "DUPLICATE_ARCHIVED"
            result["routing"] = "IGNORE_DUPLICATE"
            result["reasons"].append(f"{booklet_id} merupakan salinan duplikat dari pemindaian root/subfolder.")
            result["gates"]["gate_0_iqg"] = {"status": "SKIP_DUPLICATE"}
            return result

        gate0_pass = True
        gate0_pages = []
        if booklet_id in REJECTED_BOOKLET_IDS:
            gate0_pass = False
            result["reasons"].append(f"{booklet_id} masuk dalam daftar penolakan kualitas scan pindaian rusak (REJECT).")

        for p in range(1, 8):
            clean_file = pages_dir / f"hal_{p}_clean.png"
            if not clean_file.exists():
                gate0_pass = False
                gate0_pages.append({"page": p, "status": "FILE_NOT_FOUND"})
                continue
            im_gray = cv2.imread(str(clean_file), cv2.IMREAD_GRAYSCALE)
            if im_gray is None:
                gate0_pass = False
                gate0_pages.append({"page": p, "status": "UNREADABLE"})
                continue

            # 1. Cek rasio tinta (Blank/Washed-out rejection)
            ink_pct = float(np.mean(im_gray < 140) * 100)
            if ink_pct < 0.4:
                gate0_pass = False
                gate0_pages.append({"page": p, "status": "BLANK_WASHED_OUT", "ink_pct": round(ink_pct, 2)})
                result["reasons"].append(f"Halaman {p} pudar/putih polos tanpa tinta ({ink_pct:.2f}% < 0.40%).")
                continue

            # 2. Cek border black margin (Scanner roller shadow / misfeed rejection)
            b_mask = np.zeros_like(im_gray, dtype=bool)
            b_mask[:50, :] = True
            b_mask[-50:, :] = True
            b_mask[:, :50] = True
            b_mask[:, -50:] = True
            border_black_pct = float(np.mean(im_gray[b_mask] < 50) * 100)
            if border_black_pct > 15.0:
                gate0_pass = False
                gate0_pages.append({"page": p, "status": "HEAVY_BORDER_NOISE", "border_black_pct": round(border_black_pct, 1)})
                result["reasons"].append(f"Halaman {p} memiliki noise border hitam ekstrem ({border_black_pct:.1f}% > 15.0%).")
                continue

            gate0_pages.append({"page": p, "status": "PASS", "ink_pct": round(ink_pct, 1), "border_black_pct": round(border_black_pct, 1)})

        result["gates"]["gate_0_iqg"] = {
            "status": "PASS" if gate0_pass else "FAIL",
            "pages": gate0_pages
        }

        if not gate0_pass:
            result["decision"] = "REJECT_POOR_SCAN"
            result["routing"] = "RESCAN_QUEUE"
            return result

        # ----------------------------------------------------
        # GATE 1: Preprocessing & Affine Alignment
        # ----------------------------------------------------
        gate1_pass = True
        gate1_pages = []
        max_aspect_dev = 0.0

        b_overlay_sum = self.overlays_dir / booklet_id / "clean" / "scan_overlay_summary.json"
        cached_overlay_data = None
        if b_overlay_sum.exists() and not check_snapping:
            try:
                with open(b_overlay_sum, "r", encoding="utf-8") as f:
                    cached_overlay_data = json.load(f)
            except Exception:
                cached_overlay_data = None

        if cached_overlay_data and "pages" in cached_overlay_data:
            for p_info in cached_overlay_data["pages"]:
                p = p_info.get("page")
                dev = p_info.get("aspect_dev_pct", 0.0)
                max_aspect_dev = max(max_aspect_dev, dev)
                p_status = "PASS" if dev < 5.0 and p_info.get("status") in ["PASS", "WARN"] else "FAIL"
                if p_status == "FAIL":
                    gate1_pass = False
                    result["reasons"].append(f"Halaman {p} deviasi aspek tinggi ({dev:.2f}% >= 5.0%).")
                gate1_pages.append({
                    "page": p,
                    "dims": p_info.get("dims", []),
                    "scale_x": p_info.get("scale_x", 0.0),
                    "scale_y": p_info.get("scale_y", 0.0),
                    "aspect_dev_pct": dev,
                    "status": p_status
                })
        else:
            for p in range(1, 8):
                clean_file = pages_dir / f"hal_{p}_clean.png"
                if not clean_file.exists():
                    gate1_pass = False
                    gate1_pages.append({"page": p, "status": "FILE_NOT_FOUND"})
                    result["reasons"].append(f"Halaman {p} clean scan tidak ditemukan.")
                    continue

                img = cv2.imread(str(clean_file))
                if img is None:
                    gate1_pass = False
                    gate1_pages.append({"page": p, "status": "INVALID_IMAGE"})
                    result["reasons"].append(f"Halaman {p} berkas citra rusak/tidak terbaca.")
                    continue

                H, W = img.shape[:2]
                try:
                    align = align_template_to_image(img, page_num=p, template_dir=self.templates_dir)
                    sx, sy = align.scale_x, align.scale_y
                    aspect_dev = abs(sy - sx) / sx * 100.0
                    max_aspect_dev = max(max_aspect_dev, aspect_dev)

                    p_status = "PASS" if aspect_dev < 5.0 else "WARN_ASPECT"
                    if aspect_dev >= 5.0:
                        gate1_pass = False
                        result["reasons"].append(f"Halaman {p} deviasi aspek tinggi ({aspect_dev:.2f}% >= 5.0%).")

                    gate1_pages.append({
                        "page": p,
                        "dims": [W, H],
                        "scale_x": round(sx, 4),
                        "scale_y": round(sy, 4),
                        "aspect_dev_pct": round(aspect_dev, 2),
                        "status": p_status
                    })
                except Exception as e:
                    gate1_pass = False
                    gate1_pages.append({"page": p, "status": "ALIGN_ERROR", "error": str(e)})
                    result["reasons"].append(f"Halaman {p} gagal perataan affine: {e}")

        result["gates"]["gate_1_alignment"] = {
            "status": "PASS" if gate1_pass else "FAIL",
            "max_aspect_dev_pct": round(max_aspect_dev, 2),
            "pages": gate1_pages
        }

        # ----------------------------------------------------
        # GATE 2: Zero-Loss Digit Count Invariant (CRITICAL)
        # N_crops harus PERSIS 2.073 kotak digit (2065 kuesioner + 1 tw + 7 hal)
        # ----------------------------------------------------
        gate2_pass = True
        gate2_breakdown = {}
        total_crops_found = 0
        total_q_found = 0
        total_tw_found = 0
        total_hal_found = 0

        for p in range(1, 8):
            p_crop_raw = c_dir / f"hal_{p}" / "digits" / "raw"
            exp_p = PAGE_TARGETS[p]["total"]
            exp_q = PAGE_TARGETS[p]["questionnaire"]
            exp_tw = PAGE_TARGETS[p]["triwulan"]
            exp_hal = PAGE_TARGETS[p]["page_number"]

            if not p_crop_raw.exists():
                gate2_pass = False
                act_p = 0
                act_q = 0
                act_tw = 0
                act_hal = 0
            else:
                all_raw_files = list(p_crop_raw.rglob("*.png"))
                act_p = len(all_raw_files)
                # Hitung berdasarkan folder
                act_hal = len(list((p_crop_raw / f"HAL_{p}").glob("*.png"))) if (p_crop_raw / f"HAL_{p}").exists() else 0
                act_tw = len(list((p_crop_raw / "TRIWULAN").glob("*.png"))) if (p_crop_raw / "TRIWULAN").exists() else 0
                act_q = act_p - act_hal - act_tw

            total_crops_found += act_p
            total_q_found += act_q
            total_tw_found += act_tw
            total_hal_found += act_hal

            p_ok = (act_p == exp_p and act_q == exp_q and act_tw == exp_tw and act_hal == exp_hal)
            if not p_ok:
                gate2_pass = False
                result["reasons"].append(f"Halaman {p} defisit crop: ditemukan {act_p}/{exp_p} kotak.")

            gate2_breakdown[f"hal_{p}"] = {
                "expected": exp_p,
                "actual": act_p,
                "questionnaire": {"expected": exp_q, "actual": act_q},
                "triwulan": {"expected": exp_tw, "actual": act_tw},
                "page_number": {"expected": exp_hal, "actual": act_hal},
                "is_complete": p_ok
            }

        result["gates"]["gate_2_digit_count"] = {
            "status": "PASS" if gate2_pass else "FAIL",
            "total_crops_expected": TOTAL_DIGITS_TARGET,
            "total_crops_actual": total_crops_found,
            "completion_pct": round(total_crops_found / TOTAL_DIGITS_TARGET * 100.0, 2),
            "breakdown": {
                "questionnaire": {"expected": TOTAL_QUESTIONNAIRE_TARGET, "actual": total_q_found},
                "triwulan": {"expected": TOTAL_TRIWULAN_TARGET, "actual": total_tw_found},
                "page_number": {"expected": TOTAL_PAGE_NUMBER_TARGET, "actual": total_hal_found}
            },
            "per_page": gate2_breakdown
        }

        # ----------------------------------------------------
        # GATE 3: Substantive Snapping Quality (Pemisahan No Halaman & Triwulan)
        # ----------------------------------------------------
        # Cek apakah sudah ada hasil batch_robustness_summary atau hitung langsung
        gate3_pass = True
        snapping_metrics = {
            "questionnaire": {"total": TOTAL_QUESTIONNAIRE_TARGET, "snapped": 0},
            "triwulan": {"total": TOTAL_TRIWULAN_TARGET, "snapped": 0},
            "page_number": {"total": TOTAL_PAGE_NUMBER_TARGET, "snapped": 0}
        }

        # Cek scan_overlay_summary.json di booklet jika ada
        b_overlay_sum = self.overlays_dir / booklet_id / "clean" / "scan_overlay_summary.json"
        if b_overlay_sum.exists() and not check_snapping:
            with open(b_overlay_sum, "r", encoding="utf-8") as f:
                o_data = json.load(f)
            total_snapped_all = o_data.get("total_snapped_digits", 0)
            # Estimasi breakdown
            snapping_metrics["questionnaire"]["snapped"] = min(TOTAL_QUESTIONNAIRE_TARGET, total_snapped_all)
            q_pct = round(snapping_metrics["questionnaire"]["snapped"] / TOTAL_QUESTIONNAIRE_TARGET * 100.0, 2)
        elif check_snapping and gate1_pass:
            # Hitung snapping presisi per role secara langsung
            for p in range(1, 8):
                clean_file = pages_dir / f"hal_{p}_clean.png"
                img = cv2.imread(str(clean_file))
                align = align_template_to_image(img, page_num=p, template_dir=self.templates_dir)
                align.enable_physical_cell_snapping(img)
                m = align.get_snapping_metrics_by_role()

                snapping_metrics["questionnaire"]["snapped"] += m.get("questionnaire_digit", {}).get("snapped", 0)
                snapping_metrics["triwulan"]["snapped"] += m.get("triwulan", {}).get("snapped", 0)
                snapping_metrics["page_number"]["snapped"] += m.get("page_number", {}).get("snapped", 0)

            q_pct = round(snapping_metrics["questionnaire"]["snapped"] / TOTAL_QUESTIONNAIRE_TARGET * 100.0, 2)
        else:
            q_pct = 0.0

        q_status = "PASS" if q_pct >= 95.0 else "WARN_LOW_CONTRAST"
        if q_pct < 95.0:
            result["reasons"].append(f"Snapping kisi fisik kuesioner rendah ({q_pct:.2f}% < 95.0%). Citra pudar (menggunakan fallback affine).")

        result["gates"]["gate_3_snapping_quality"] = {
            "status": q_status,
            "questionnaire_snapping_pct": q_pct,
            "metrics_by_role": {
                "questionnaire_substantive": {
                    "total": TOTAL_QUESTIONNAIRE_TARGET,
                    "snapped": snapping_metrics["questionnaire"]["snapped"],
                    "pct": q_pct
                },
                "triwulan_administrative": {
                    "total": TOTAL_TRIWULAN_TARGET,
                    "snapped": snapping_metrics["triwulan"]["snapped"],
                    "pct": round(snapping_metrics["triwulan"]["snapped"] / TOTAL_TRIWULAN_TARGET * 100.0, 2)
                },
                "page_number_footer": {
                    "total": TOTAL_PAGE_NUMBER_TARGET,
                    "snapped": snapping_metrics["page_number"]["snapped"],
                    "pct": round(snapping_metrics["page_number"]["snapped"] / TOTAL_PAGE_NUMBER_TARGET * 100.0, 2)
                }
            }
        }

        # ----------------------------------------------------
        # GATE 4: Production Decision & Routing
        # ----------------------------------------------------
        if not gate1_pass or not gate2_pass:
            decision = "REJECT_INCOMPLETE"
            routing = "STOP_AND_FIX"
        elif q_status == "WARN_LOW_CONTRAST":
            decision = "FLAG_FOR_HITL_REVIEW"
            routing = "HITL_VISUAL_AUDIT_QUEUE"
        else:
            decision = "PROCEED_TO_OCR"
            routing = "AUTO_OCR_PIPELINE"

        result["gates"]["gate_4_decision"] = {
            "decision": decision,
            "routing": routing
        }
        result["decision"] = decision
        result["routing"] = routing

        return result

    def validate_all_booklets(self, check_snapping_deep: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Audit seluruh buklet di workspace dan hasilkan laporan konsolidasi.
        """
        all_bids = sorted([d.name for d in self.booklets_dir.iterdir() if d.is_dir() and d.name.startswith("booklet_")])
        if limit:
            all_bids = all_bids[:limit]

        start_time = time.time()
        results = []
        counts = {
            "PROCEED_TO_OCR": 0,
            "FLAG_FOR_HITL_REVIEW": 0,
            "REJECT_INCOMPLETE": 0
        }

        for bid in all_bids:
            res = self.validate_booklet(bid, check_snapping=check_snapping_deep)
            results.append(res)
            d = res["decision"]
            counts[d] = counts.get(d, 0) + 1

        elapsed = round(time.time() - start_time, 2)
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_booklets": len(results),
            "elapsed_seconds": elapsed,
            "decisions": counts,
            "pass_rate_pct": round(counts["PROCEED_TO_OCR"] / len(results) * 100.0, 2) if results else 0,
            "hitl_review_count": counts["FLAG_FOR_HITL_REVIEW"],
            "reject_count": counts["REJECT_INCOMPLETE"],
            "booklets": results
        }

        # Simpan laporan resmi
        reports_dir = self.workspace / "output" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / "stage_gate_audit_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
