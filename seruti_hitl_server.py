#!/usr/bin/env python3
"""
seruti_hitl_server.py
=====================
Server Web Interaktif Human-in-the-Loop (HITL) untuk Koreksi Anotasi Hasil Auto-Labeling Seruti.
Tema Visual: "Seashell garnet afternoon"
Palet Warna:
  - Seashell Sand: #F6C992
  - Garnet Coral : #F96E81
  - Afternoon Mist : #ACC0D3
  - Dusty Mauve  : #D396A6
  - Deep Teal    : #09A1A1
  - Slate Steel  : #5484A4
  - Pine Slate   : #30525C

Fokus pada Opsi A: 721 Citra Prioritas Tinggi (Tier 1: Ambigu Fisik & Tier 2: Konfidensi <= 0.85).
"""

import os
import sys
import json
import base64
import shutil
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import socketserver

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True
from urllib.parse import urlparse, parse_qs
import pandas as pd

CSV_PATH = "hasil_labeling/hasil_autolabel.csv"
LOG_PATH = "hasil_labeling/manual_corrections_log.jsonl"
BASE_IMAGE_DIR = "hasil_labeling"
PORT = 8088

# State Manajer
class HITLManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = []
        self.corrections = {} # nama_file_unik -> {new_label, status, timestamp}
        self.history = []     # stack of actions for undo
        self.pending_csv_updates = {}
        self.dirty = False
        self.last_saved_time = datetime.now()
        self.save_lock = threading.Lock()
        self.load_data()
        self._start_autosave_worker()

    def _start_autosave_worker(self):
        import time
        def worker():
            while True:
                time.sleep(2.0)
                if self.dirty:
                    self.flush_to_disk()
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def flush_to_disk(self):
        with self.save_lock:
            with self.lock:
                if not self.dirty or not self.pending_csv_updates:
                    return
                batch = dict(self.pending_csv_updates)
                self.pending_csv_updates.clear()
                self.dirty = False

            try:
                df = pd.read_csv(CSV_PATH)
                mask = df["nama_file_unik"].isin(batch.keys())
                if mask.any():
                    df.loc[mask, "prediksi_label"] = df.loc[mask, "nama_file_unik"].map(batch)
                    df.to_csv(CSV_PATH, index=False)
                    self.all_df = df
                    shutil.copy2(CSV_PATH, "hasil_labeling/hasil_autolabel_autosave.csv")
                self.last_saved_time = datetime.now()
                print(f"[*] Auto-Save: {len(batch)} koreksi berhasil disimpan ke CSV pada {self.last_saved_time.strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"[-] Auto-Save gagal: {e}")
                with self.lock:
                    for k, v in batch.items():
                        if k not in self.pending_csv_updates:
                            self.pending_csv_updates[k] = v
                    self.dirty = True

    def load_data(self):
        with self.lock:
            if not os.path.exists(CSV_PATH):
                print(f"[-] CSV {CSV_PATH} tidak ditemukan!")
                return

            df = pd.read_csv(CSV_PATH)
            self.all_df = df

            # Muat log riwayat koreksi jika ada
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                fn = entry.get("nama_file_unik")
                                if fn:
                                    self.corrections[fn] = entry
                            except Exception:
                                pass

            # Filter Opsi A: Tier 1 (ragu_ragu) + Tier 2 (digits conf <= 0.85)
            t1 = df[df["prediksi_label"] == "ragu_ragu"].copy()
            t1["tier"] = 1
            t1["alasan"] = "Tier 1: Ambigu Fisik / Cacat Lapangan"

            t2 = df[(df["prediksi_label"] != "ragu_ragu") & 
                    (df["prediksi_label"] != "kosong") & 
                    (df["confidence"] <= 0.85)].copy()
            t2["tier"] = 2
            t2["alasan"] = "Tier 2: Konfidensi Rendah (<= 0.85)"

            queue_df = pd.concat([t1, t2]).reset_index(drop=True)
            print(f"[*] Berhasil memuat antrean Opsi A: {len(queue_df)} citra.")

            items = []
            for idx, r in queue_df.iterrows():
                fn = str(r["nama_file_unik"])
                orig_label = str(r["prediksi_label"])
                conf = float(r.get("confidence", 1.0))
                tier = int(r.get("tier", 1))
                alasan = str(r.get("alasan", ""))
                booklet = str(r.get("id_booklet", ""))
                halaman = str(r.get("halaman", ""))
                field_id = str(r.get("field_id", ""))
                box_id = str(r.get("box_id", ""))

                # Cek koreksi tersimpan
                current_label = orig_label
                status = "unreviewed" # unreviewed | approved | corrected
                reviewed_at = None

                if fn in self.corrections:
                    corr = self.corrections[fn]
                    current_label = str(corr.get("new_label", orig_label))
                    status = corr.get("status", "corrected")
                    reviewed_at = corr.get("timestamp")

                # Muat citra dan konversi base64
                img_path = os.path.join(BASE_IMAGE_DIR, current_label, fn)
                if not os.path.exists(img_path):
                    img_path = os.path.join(BASE_IMAGE_DIR, orig_label, fn)

                img_b64 = ""
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_f:
                        img_b64 = base64.b64encode(img_f.read()).decode("utf-8")

                items.append({
                    "id": idx,
                    "nama_file_unik": fn,
                    "orig_label": orig_label,
                    "current_label": current_label,
                    "confidence": conf,
                    "tier": tier,
                    "alasan": alasan,
                    "id_booklet": booklet,
                    "halaman": halaman,
                    "field_id": field_id,
                    "box_id": box_id,
                    "status": status,
                    "reviewed_at": reviewed_at,
                    "image_b64": f"data:image/png;base64,{img_b64}" if img_b64 else ""
                })

            self.items = items

    def get_stats(self):
        with self.lock:
            total = len(self.items)
            unreviewed = sum(1 for it in self.items if it["status"] == "unreviewed")
            approved = sum(1 for it in self.items if it["status"] == "approved")
            corrected = sum(1 for it in self.items if it["status"] == "corrected")
            tier1_count = sum(1 for it in self.items if it["tier"] == 1)
            tier2_count = sum(1 for it in self.items if it["tier"] == 2)
            return {
                "total": total,
                "unreviewed": unreviewed,
                "approved": approved,
                "corrected": corrected,
                "reviewed": approved + corrected,
                "tier1_total": tier1_count,
                "tier2_total": tier2_count,
                "progress_pct": round(((approved + corrected) / total * 100), 1) if total > 0 else 0,
                "dirty": self.dirty,
                "pending_count": len(self.pending_csv_updates),
                "last_saved": self.last_saved_time.strftime("%H:%M:%S")
            }

    def save_action(self, fn: str, new_label: str, is_approval: bool = False, operator: str = "Operator BPS"):
        with self.lock:
            target_item = None
            for it in self.items:
                if it["nama_file_unik"] == fn:
                    target_item = it
                    break

            if not target_item:
                return {"success": False, "error": f"Item {fn} tidak ditemukan"}

            old_label = target_item["current_label"]
            status = "approved" if is_approval else "corrected"
            now_iso = datetime.now().isoformat()

            # Record history for undo
            self.history.append({
                "nama_file_unik": fn,
                "prev_label": old_label,
                "prev_status": target_item["status"],
                "prev_reviewed_at": target_item["reviewed_at"]
            })

            # Pindahkan file fisik jika label berubah
            if old_label != new_label:
                src_path = os.path.join(BASE_IMAGE_DIR, old_label, fn)
                dst_dir = os.path.join(BASE_IMAGE_DIR, new_label)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, fn)

                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)

            # Update item memory
            target_item["current_label"] = new_label
            target_item["status"] = status
            target_item["reviewed_at"] = now_iso

            # Simpan log
            log_entry = {
                "timestamp": now_iso,
                "operator": operator,
                "nama_file_unik": fn,
                "old_label": old_label,
                "new_label": new_label,
                "status": status,
                "is_approval": is_approval,
                "tier": target_item["tier"]
            }
            self.corrections[fn] = log_entry

            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Tandai untuk background Auto-Save
            self.pending_csv_updates[fn] = new_label
            self.dirty = True

            return {
                "success": True,
                "nama_file_unik": fn,
                "current_label": new_label,
                "status": status,
                "stats": self.get_stats()
            }

    def get_grid_items(self, label="1", page=1, limit=100, conf_filter="all", booklet="all", halaman="all", sort="conf_asc", search=""):
        with self.lock:
            if not hasattr(self, "all_df") or self.all_df is None:
                self.all_df = pd.read_csv(CSV_PATH)
            df = self.all_df

        # Filter by label
        if label == "all_digits":
            sub = df[df["prediksi_label"].isin([str(i) for i in range(10)])]
        elif label == "romawi":
            sub = df[df["prediksi_label"].isin(["I", "II", "III", "IV"])]
        elif label == "all":
            sub = df
        else:
            sub = df[df["prediksi_label"] == label]

        # Filter confidence
        if conf_filter == "low":
            sub = sub[sub["confidence"] <= 0.85]
        elif conf_filter == "med":
            sub = sub[sub["confidence"] <= 0.90]
        elif conf_filter == "high":
            sub = sub[sub["confidence"] > 0.85]

        # Filter booklet
        if booklet != "all":
            sub = sub[sub["id_booklet"] == booklet]

        # Filter halaman
        if halaman != "all":
            sub = sub[sub["halaman"] == halaman]

        # Search
        if search:
            s = search.lower().strip()
            sub = sub[
                sub["nama_file_unik"].str.lower().str.contains(s) |
                sub["field_id"].str.lower().str.contains(s) |
                sub["id_booklet"].str.lower().str.contains(s) |
                sub["box_id"].str.lower().str.contains(s)
            ]

        # Sort
        if sort == "conf_asc":
            sub = sub.sort_values(by="confidence", ascending=True)
        elif sort == "conf_desc":
            sub = sub.sort_values(by="confidence", ascending=False)
        elif sort == "booklet":
            sub = sub.sort_values(by=["id_booklet", "halaman", "field_id"])

        total_items = len(sub)
        total_pages = max(1, (total_items + limit - 1) // limit)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        page_df = sub.iloc[start_idx:end_idx]

        # Global label counts from master df
        counts = df["prediksi_label"].value_counts().to_dict()

        records = []
        for _, r in page_df.iterrows():
            fn = str(r["nama_file_unik"])
            curr_lbl = str(r["prediksi_label"])
            records.append({
                "nama_file_unik": fn,
                "id_booklet": str(r["id_booklet"]),
                "halaman": str(r["halaman"]),
                "field_id": str(r["field_id"]),
                "box_id": str(r["box_id"]),
                "prediksi_label": curr_lbl,
                "current_label": curr_lbl,
                "confidence": round(float(r.get("confidence", 1.0)), 2)
            })

        return {
            "total_items": total_items,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "items": records,
            "counts_by_label": counts
        }

    def bulk_correct(self, updates: list, operator: str = "Operator BPS"):
        with self.lock:
            if not hasattr(self, "all_df") or self.all_df is None:
                self.all_df = pd.read_csv(CSV_PATH)

            now_iso = datetime.now().isoformat()
            undo_items = []
            updated_count = 0

            for it in updates:
                fn = it.get("nama_file_unik")
                new_lbl = str(it.get("new_label"))
                old_lbl = it.get("old_label")

                if not fn or not new_lbl:
                    continue

                if not old_lbl:
                    match = self.all_df[self.all_df["nama_file_unik"] == fn]
                    if not match.empty:
                        old_lbl = str(match.iloc[0]["prediksi_label"])
                    else:
                        old_lbl = new_lbl

                if old_lbl != new_lbl:
                    src_path = os.path.join(BASE_IMAGE_DIR, old_lbl, fn)
                    dst_dir = os.path.join(BASE_IMAGE_DIR, new_lbl)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_path = os.path.join(dst_dir, fn)

                    if os.path.exists(src_path):
                        try:
                            shutil.move(src_path, dst_path)
                        except Exception as e:
                            print(f"[-] Gagal memindahkan file {fn}: {e}")

                    mask = self.all_df["nama_file_unik"] == fn
                    if mask.any():
                        self.all_df.loc[mask, "prediksi_label"] = new_lbl

                    for q_item in self.items:
                        if q_item["nama_file_unik"] == fn:
                            q_item["current_label"] = new_lbl
                            q_item["status"] = "corrected"
                            q_item["reviewed_at"] = now_iso
                            break

                    self.pending_csv_updates[fn] = new_lbl
                    self.dirty = True

                    log_entry = {
                        "timestamp": now_iso,
                        "operator": operator,
                        "nama_file_unik": fn,
                        "old_label": old_lbl,
                        "new_label": new_lbl,
                        "status": "bulk_corrected",
                        "is_approval": False,
                        "tier": "bulk_grid"
                    }
                    self.corrections[fn] = log_entry
                    try:
                        with open(LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(log_entry) + "\n")
                    except Exception:
                        pass

                    undo_items.append({
                        "nama_file_unik": fn,
                        "prev_label": old_lbl,
                        "new_label": new_lbl
                    })
                    updated_count += 1

            if undo_items:
                self.history.append({"type": "bulk", "items": undo_items})

            if self.dirty:
                self.flush_to_disk()

            return {"success": True, "updated": updated_count}

    def undo_action(self):
        with self.lock:
            if not self.history:
                return {"success": False, "error": "Tidak ada riwayat koreksi untuk dibatalkan"}

            last = self.history.pop()

            if last.get("type") == "bulk":
                reverted_items = []
                for item in last["items"]:
                    fn = item["nama_file_unik"]
                    prev_lbl = item["prev_label"]
                    curr_lbl = item["new_label"]

                    src_path = os.path.join(BASE_IMAGE_DIR, curr_lbl, fn)
                    dst_dir = os.path.join(BASE_IMAGE_DIR, prev_lbl)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_path = os.path.join(dst_dir, fn)
                    if os.path.exists(src_path):
                        shutil.move(src_path, dst_path)

                    mask = self.all_df["nama_file_unik"] == fn
                    if mask.any():
                        self.all_df.loc[mask, "prediksi_label"] = prev_lbl

                    for q_item in self.items:
                        if q_item["nama_file_unik"] == fn:
                            q_item["current_label"] = prev_lbl
                            break

                    if fn in self.corrections:
                        del self.corrections[fn]

                    self.pending_csv_updates[fn] = prev_lbl
                    self.dirty = True
                    reverted_items.append(fn)

                self.flush_to_disk()
                return {
                    "success": True,
                    "message": f"Berhasil membatalkan koreksi {len(reverted_items)} citra",
                    "reverted_count": len(reverted_items)
                }

            fn = last["nama_file_unik"]
            prev_label = last["prev_label"]
            prev_status = last["prev_status"]

            target_item = None
            for it in self.items:
                if it["nama_file_unik"] == fn:
                    target_item = it
                    break

            if not target_item:
                return {"success": False, "error": "Item tidak ditemukan"}

            curr_label = target_item["current_label"]

            # Balikkan perpindahan file
            if curr_label != prev_label:
                src_path = os.path.join(BASE_IMAGE_DIR, curr_label, fn)
                dst_dir = os.path.join(BASE_IMAGE_DIR, prev_label)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, fn)
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)

            target_item["current_label"] = prev_label
            target_item["status"] = prev_status
            target_item["reviewed_at"] = last["prev_reviewed_at"]

            if fn in self.corrections:
                del self.corrections[fn]

            mask = self.all_df["nama_file_unik"] == fn
            if mask.any():
                self.all_df.loc[mask, "prediksi_label"] = prev_label

            self.pending_csv_updates[fn] = prev_label
            self.dirty = True

            return {
                "success": True,
                "nama_file_unik": fn,
                "current_label": prev_label,
                "status": prev_status,
                "stats": self.get_stats()
            }


DATA_MGR = HITLManager()

HTML_PAGE = """<!DOCTYPE html>
<html lang="id" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BPS Seruti - HITL Verifier | Seashell Garnet Afternoon</title>
<style>
/* ==========================================================================
   PALET WARNA: "Seashell garnet afternoon"
   Swatches:
     --c-seashell : #F6C992 (Warm Sand / Apricot Peach)
     --c-garnet   : #F96E81 (Vibrant Coral Pink / Garnet)
     --c-mist     : #ACC0D3 (Afternoon Sky / Slate Mist)
     --c-mauve    : #D396A6 (Dusty Rose Mauve)
     --c-teal     : #09A1A1 (Deep Vibrant Teal Cyan)
     --c-steel    : #5484A4 (Slate Steel Blue)
     --c-pine     : #30525C (Deep Slate Pine / Charcoal)
   ========================================================================== */

:root {
  /* Core Swatches */
  --c-seashell: #F6C992;
  --c-garnet: #F96E81;
  --c-mist: #ACC0D3;
  --c-mauve: #D396A6;
  --c-teal: #09A1A1;
  --c-steel: #5484A4;
  --c-pine: #30525C;
}

[data-theme="dark"] {
  --bg-app: #18252a;
  --bg-surface: #22343b;
  --bg-surface-elevated: #2a4149;
  --bg-input: #152024;
  --border-color: rgba(172, 192, 211, 0.22);
  --border-highlight: var(--c-teal);
  
  --text-primary: #FFFFFF;
  --text-secondary: var(--c-mist);
  --text-accent: var(--c-seashell);
  
  --badge-tier1-bg: rgba(246, 201, 146, 0.16);
  --badge-tier1-text: var(--c-seashell);
  --badge-tier1-border: var(--c-seashell);
  
  --badge-tier2-bg: rgba(172, 192, 211, 0.16);
  --badge-tier2-text: var(--c-mist);
  --badge-tier2-border: var(--c-mist);

  --btn-key-bg: #19272c;
  --btn-key-text: #FFFFFF;
  --btn-key-hover-bg: #2d454e;
  --btn-key-hover-border: var(--c-teal);

  --card-shadow: 0 16px 36px -8px rgba(10, 18, 21, 0.7);
  --frame-bg: #ffffff;
}

[data-theme="light"] {
  --bg-app: #f9f5ee;
  --bg-surface: #ffffff;
  --bg-surface-elevated: #f4ecdf;
  --bg-input: #fdfbf7;
  --border-color: rgba(84, 132, 164, 0.25);
  --border-highlight: var(--c-teal);
  
  --text-primary: #1f3137;
  --text-secondary: #4a6673;
  --text-accent: #098b8b;
  
  --badge-tier1-bg: rgba(246, 201, 146, 0.35);
  --badge-tier1-text: #965d1b;
  --badge-tier1-border: #e09f54;
  
  --badge-tier2-bg: rgba(172, 192, 211, 0.35);
  --badge-tier2-text: #2f4b5e;
  --badge-tier2-border: #7395ad;

  --btn-key-bg: #f8f4ec;
  --btn-key-text: #1f3137;
  --btn-key-hover-bg: #efe5d5;
  --btn-key-hover-border: var(--c-teal);

  --card-shadow: 0 16px 32px -8px rgba(48, 82, 92, 0.15);
  --frame-bg: #ffffff;
}

* { box-sizing: border-box; margin: 0; padding: 0; transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Plus Jakarta Sans', sans-serif;
  background-color: var(--bg-app);
  color: var(--text-primary);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header */
header {
  background: var(--bg-surface);
  border-bottom: 2px solid var(--border-color);
  padding: 14px 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.brand-section {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-palette-dots {
  display: flex;
  gap: 4px;
}
.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.dot-1 { background: var(--c-seashell); }
.dot-2 { background: var(--c-garnet); }
.dot-3 { background: var(--c-mist); }
.dot-4 { background: var(--c-mauve); }
.dot-5 { background: var(--c-teal); }
.dot-6 { background: var(--c-steel); }

.brand-title {
  font-size: 18px;
  font-weight: 800;
  color: var(--text-primary);
  letter-spacing: 0.5px;
}

.brand-tag {
  background: var(--badge-tier1-bg);
  color: var(--badge-tier1-text);
  border: 1px solid var(--badge-tier1-border);
  font-size: 11px;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 999px;
  text-transform: uppercase;
}

/* Stats Ribbon */
.stats-container {
  display: flex;
  align-items: center;
  gap: 14px;
}

.prog-container {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  font-weight: 600;
}

.prog-track {
  width: 130px;
  height: 10px;
  background: rgba(172, 192, 211, 0.2);
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid var(--border-color);
}

.prog-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--c-teal), var(--c-seashell), var(--c-garnet));
  width: 0%;
  transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stat-chip {
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--bg-surface-elevated);
  border: 1px solid var(--border-color);
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
}

.theme-toggle-btn {
  background: var(--bg-surface-elevated);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
}
.theme-toggle-btn:hover {
  border-color: var(--c-teal);
  color: var(--c-teal);
}

/* Main Workspace */
main {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px 20px;
  max-width: 860px;
  margin: 0 auto;
  width: 100%;
}

/* Review Card */
.review-card {
  background: var(--bg-surface);
  border: 2px solid var(--border-color);
  border-radius: 20px;
  width: 100%;
  padding: 32px 36px;
  box-shadow: var(--card-shadow);
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
}

.top-pills {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}

.tier-pill {
  font-size: 12px;
  font-weight: 700;
  padding: 5px 12px;
  border-radius: 8px;
}
.tier-1 {
  background: var(--badge-tier1-bg);
  color: var(--badge-tier1-text);
  border: 1px solid var(--badge-tier1-border);
}
.tier-2 {
  background: var(--badge-tier2-bg);
  color: var(--badge-tier2-text);
  border: 1px solid var(--badge-tier2-border);
}

.status-pill {
  font-size: 12px;
  font-weight: 700;
  padding: 5px 12px;
  border-radius: 8px;
}
.st-unreviewed {
  background: rgba(84, 132, 164, 0.2);
  color: var(--c-steel);
  border: 1px solid var(--c-steel);
}
.st-approved {
  background: rgba(9, 161, 161, 0.2);
  color: var(--c-teal);
  border: 1px solid var(--c-teal);
}
.st-corrected {
  background: rgba(249, 110, 129, 0.2);
  color: var(--c-garnet);
  border: 1px solid var(--c-garnet);
}

/* Image Display Frame */
.image-stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 10px 0 16px;
}

.crop-frame {
  width: 195px;
  height: 195px;
  background: var(--frame-bg);
  border-radius: 14px;
  border: 3px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 8px 24px rgba(0,0,0,0.18);
  overflow: hidden;
  transition: transform 0.15s, border-color 0.2s;
}
.crop-frame:hover {
  border-color: var(--c-teal);
  transform: scale(1.02);
}
.crop-frame.invert-mode {
  filter: invert(1) hue-rotate(180deg);
  background: #000000;
}
.crop-frame img {
  max-width: 180px;
  max-height: 180px;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}

.btn-invert {
  margin-top: 10px;
  background: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 6px;
  cursor: pointer;
}
.btn-invert:hover {
  color: var(--text-primary);
  border-color: var(--c-mist);
}

/* Label & Metadata Banner */
.label-info-banner {
  text-align: center;
  margin-bottom: 20px;
}

.current-label-display {
  font-size: 32px;
  font-weight: 900;
  color: var(--c-seashell);
  letter-spacing: 0.5px;
  text-shadow: 0 2px 10px rgba(246, 201, 146, 0.25);
}

.meta-field-tag {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-top: 4px;
}

.meta-conf-tag {
  font-size: 12px;
  font-weight: 600;
  color: var(--c-mist);
  margin-top: 3px;
}

/* Keypad Grid 0-9 */
.keypad-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  width: 100%;
  max-width: 520px;
  margin-bottom: 14px;
}

.key-btn {
  background: var(--btn-key-bg);
  border: 1.5px solid var(--border-color);
  border-radius: 10px;
  color: var(--btn-key-text);
  padding: 12px 6px;
  font-size: 18px;
  font-weight: 800;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.06);
  transition: all 0.12s ease;
}
.key-btn:hover, .key-btn:focus {
  background: var(--btn-key-hover-bg);
  border-color: var(--btn-key-hover-border);
  color: var(--c-teal);
  transform: translateY(-2px);
  box-shadow: 0 6px 14px rgba(9, 161, 161, 0.2);
}
.key-btn:active {
  transform: translateY(0);
}
.key-sub {
  font-size: 10px;
  color: var(--text-secondary);
  font-weight: 500;
}

/* Action Buttons (Approve, Kosong, Centang, Ragu) */
.action-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  width: 100%;
  max-width: 520px;
  margin-bottom: 20px;
}

.action-btn {
  padding: 12px 8px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 800;
  border: none;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  box-shadow: 0 4px 10px rgba(0,0,0,0.12);
  transition: all 0.12s ease;
}
.action-btn:hover {
  transform: translateY(-2px);
  filter: brightness(1.1);
}
.action-btn:active {
  transform: translateY(0);
}

.btn-approve {
  background: var(--c-teal);
  color: #FFFFFF;
}
.btn-kosong {
  background: var(--c-steel);
  color: #FFFFFF;
}
.btn-centang {
  background: var(--c-mauve);
  color: #FFFFFF;
}
.btn-ragu {
  background: var(--c-garnet);
  color: #FFFFFF;
}

/* Navigation Row */
.nav-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  max-width: 520px;
  padding-top: 16px;
  border-top: 1px solid var(--border-color);
}

.nav-btn {
  background: var(--bg-surface-elevated);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}
.nav-btn:hover {
  border-color: var(--c-teal);
  color: var(--c-teal);
}

.jump-group {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
}
.jump-input {
  background: var(--bg-input);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 6px 10px;
  border-radius: 6px;
  width: 72px;
  text-align: center;
  font-weight: 700;
}

/* Hotkey Legend */
.hotkey-legend {
  margin-top: 20px;
  font-size: 11px;
  color: var(--text-secondary);
  text-align: center;
  line-height: 1.8;
}
.kbd {
  background: var(--bg-surface-elevated);
  border: 1px solid var(--border-color);
  padding: 2px 7px;
  border-radius: 4px;
  font-family: monospace;
  font-weight: 700;
  color: var(--text-primary);
}

/* Floating Matrix Drawer Button */
.matrix-fab {
  position: fixed;
  bottom: 24px;
  right: 28px;
  background: var(--c-teal);
  color: #FFFFFF;
  border: 2px solid rgba(255, 255, 255, 0.4);
  font-weight: 800;
  font-size: 13px;
  padding: 12px 22px;
  border-radius: 999px;
  cursor: pointer;
  box-shadow: 0 8px 24px rgba(9, 161, 161, 0.4);
  z-index: 99;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: transform 0.15s;
}
.matrix-fab:hover {
  transform: scale(1.05);
}

/* Drawer Modal */
.drawer-modal {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(15, 25, 29, 0.85);
  backdrop-filter: blur(4px);
  z-index: 1000;
  padding: 40px 20px;
  overflow: auto;
}
.drawer-modal.open { display: block; }
.drawer-card {
  background: var(--bg-surface);
  border: 2px solid var(--border-color);
  border-radius: 20px;
  max-width: 1120px;
  margin: 0 auto;
  padding: 28px;
  box-shadow: var(--card-shadow);
}
.drawer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.matrix-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(72px, 1fr));
  gap: 10px;
  max-height: 68vh;
  overflow-y: auto;
  padding-right: 8px;
}
.mat-card {
  background: var(--bg-surface-elevated);
  border: 1.5px solid var(--border-color);
  border-radius: 8px;
  padding: 4px;
  text-align: center;
  cursor: pointer;
  transition: transform 0.1s, border-color 0.1s;
}
.mat-card:hover {
  transform: scale(1.06);
  border-color: var(--c-teal);
}
.mat-card.c-approved { border-left: 4px solid var(--c-teal); }
.mat-card.c-corrected { border-left: 4px solid var(--c-garnet); }
.mat-card.c-ragu { border-left: 4px solid var(--c-seashell); }
.mat-card img {
  width: 100%;
  height: 48px;
  object-fit: contain;
  background: #ffffff;
  border-radius: 4px;
}
.mat-label {
  font-size: 11px;
  font-weight: 800;
  margin-top: 3px;
  color: var(--text-primary);
}

/* Optimasi Layar HP / Mobile Hotspot */
@media (max-width: 600px) {
  header { padding: 10px 14px; flex-wrap: wrap; gap: 8px; }
  .brand-title { font-size: 15px; }
  .stats-container { width: 100%; justify-content: space-between; }
  .prog-track { width: 90px; }
  main { padding: 12px 8px; }
  .review-card { padding: 20px 14px; border-radius: 14px; }
  .crop-frame { width: 170px; height: 170px; }
  .current-label-display { font-size: 26px; }
  .keypad-grid { gap: 8px; }
  .key-btn { padding: 14px 4px; font-size: 20px; touch-action: manipulation; }
  .action-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .action-btn { padding: 14px 8px; font-size: 14px; touch-action: manipulation; }
  .nav-bar { flex-wrap: wrap; gap: 10px; justify-content: center; }
  .matrix-fab { bottom: 16px; right: 16px; padding: 10px 18px; font-size: 12px; }
}
</style>
</head>
<body>

<header>
  <div class="brand-section">
    <div class="brand-palette-dots">
      <div class="dot dot-1" title="#F6C992"></div>
      <div class="dot dot-2" title="#F96E81"></div>
      <div class="dot dot-3" title="#ACC0D3"></div>
      <div class="dot dot-4" title="#D396A6"></div>
      <div class="dot dot-5" title="#09A1A1"></div>
      <div class="dot dot-6" title="#5484A4"></div>
    </div>
    <div class="brand-title">BPS SERUTI &mdash; HITL VERIFIER</div>
    <div class="brand-tag">Seashell Garnet Afternoon</div>
  </div>

  <div class="stats-container">
    <div class="prog-container">
      <div class="prog-track">
        <div class="prog-fill" id="progFill"></div>
      </div>
      <span id="statReviewed">0</span> / <span id="statTotal">721</span>
      <span id="statPct" style="color:var(--c-teal);">(0%)</span>
    </div>

    <div class="stat-chip" style="border-color:var(--c-teal); color:var(--c-teal);">
      &check; <span id="statApprove">0</span>
    </div>
    <div class="stat-chip" style="border-color:var(--c-garnet); color:var(--c-garnet);">
      &Delta; <span id="statCorrect">0</span>
    </div>

    <div class="stat-chip" id="autoSaveChip" style="border-color:var(--c-teal); color:var(--c-teal); cursor:pointer;" onclick="triggerManualSync()" title="Klik untuk sinkronisasi paksa">
      <span id="saveIcon">💾</span> <span id="saveStatus">Auto-Save Aktif</span>
    </div>

    <a href="/grid" class="stat-chip" style="border-color:var(--c-teal); color:var(--c-teal); text-decoration:none;" title="Buka Mode Koreksi Massal Grid per Angka">
      &#9889; <span>Koreksi Grid Massal</span>
    </a>

    <a href="/anomali" class="stat-chip" style="border-color:var(--c-mist); color:var(--text-primary); text-decoration:none;" title="Lihat Galeri Inspektur Anomali">
      &#128269; <span>Cek Anomali</span>
    </a>

    <button class="theme-toggle-btn" onclick="toggleTheme()">
      <span id="themeIcon">&#9790;</span> <span id="themeLabel">Dark</span>
    </button>
  </div>
</header>

<main>
  <div class="review-card">
    <div class="top-pills">
      <div class="tier-pill" id="cardTier">Tier 1: Ambigu Fisik</div>
      <div class="status-pill st-unreviewed" id="cardStatus">Belum Diperiksa</div>
    </div>

    <div class="image-stage">
      <div class="crop-frame" id="imgFrame">
        <img id="mainImg" src="" alt="Comb Box Crop">
      </div>
      <button class="btn-invert" onclick="toggleInvert()">Toggle Inversi Kertas [I]</button>
    </div>

    <div class="label-info-banner">
      <div class="current-label-display" id="currLblText">Label: <b>?</b></div>
      <div class="meta-field-tag" id="metaDetails">booklet_01 | hal_1 | field_id | box_00</div>
      <div class="meta-conf-tag" id="confReason">Konfidensi: 0.60 | Asli: 0</div>
    </div>

    <!-- Triwulan Roman Numeral Quick Row -->
    <div id="triwulanSection" style="display:none; width:100%; max-width:520px; margin-bottom:14px; background:rgba(9, 161, 161, 0.14); border:1.5px solid var(--c-teal); border-radius:12px; padding:10px 14px; text-align:center;">
      <div style="font-size:12px; font-weight:800; color:var(--c-teal); margin-bottom:8px; letter-spacing:0.5px;">PILIHAN CEPAT TRIWULAN (ANGKA ROMAWI):</div>
      <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:8px;">
        <button class="action-btn" style="background:var(--c-teal); color:#fff; font-weight:800; padding:10px 4px;" onclick="applyLabel('I')">I [1]</button>
        <button class="action-btn" style="background:var(--c-steel); color:#fff; font-weight:800; padding:10px 4px;" onclick="applyLabel('II')">II [2]</button>
        <button class="action-btn" style="background:var(--c-mauve); color:#fff; font-weight:800; padding:10px 4px;" onclick="applyLabel('III')">III [3]</button>
        <button class="action-btn" style="background:var(--c-garnet); color:#fff; font-weight:800; padding:10px 4px;" onclick="applyLabel('IV')">IV [4]</button>
      </div>
    </div>

    <!-- Keypad Grid 0-9 -->
    <div class="keypad-grid">
      <button class="key-btn" onclick="applyLabel('0')">0<span class="key-sub">[0]</span></button>
      <button class="key-btn" onclick="applyLabel('1')">1<span class="key-sub">[1]</span></button>
      <button class="key-btn" onclick="applyLabel('2')">2<span class="key-sub">[2]</span></button>
      <button class="key-btn" onclick="applyLabel('3')">3<span class="key-sub">[3]</span></button>
      <button class="key-btn" onclick="applyLabel('4')">4<span class="key-sub">[4]</span></button>
      <button class="key-btn" onclick="applyLabel('5')">5<span class="key-sub">[5]</span></button>
      <button class="key-btn" onclick="applyLabel('6')">6<span class="key-sub">[6]</span></button>
      <button class="key-btn" onclick="applyLabel('7')">7<span class="key-sub">[7]</span></button>
      <button class="key-btn" onclick="applyLabel('8')">8<span class="key-sub">[8]</span></button>
      <button class="key-btn" onclick="applyLabel('9')">9<span class="key-sub">[9]</span></button>
    </div>

    <!-- Special Action Buttons -->
    <div class="action-grid">
      <button class="action-btn btn-approve" onclick="approveCurrent()">&check; Setujui [Enter/A]</button>
      <button class="action-btn btn-kosong" onclick="applyLabel('kosong')">&#8709; Kosong [K/Spasi]</button>
      <button class="action-btn btn-centang" onclick="applyLabel('centang')">&#10003; Centang [C]</button>
      <button class="action-btn btn-ragu" onclick="applyLabel('ragu_ragu')">? Ragu/Cacat [R]</button>
    </div>

    <!-- Navigation Footer -->
    <div class="nav-bar">
      <button class="nav-btn" onclick="prevItem()">&larr; Sebelumnya [&larr;]</button>
      <div class="jump-group">
        <span>No.</span>
        <input type="number" id="jumpIdx" class="jump-input" min="1" max="721" value="1" onchange="jumpTo(this.value - 1)">
        <span id="jumpTotal">/ 721</span>
      </div>
      <button class="nav-btn" onclick="undoLast()">Undo [Ctrl+Z]</button>
      <button class="nav-btn" onclick="nextItem()">Selanjutnya [&rarr;] &rarr;</button>
    </div>

    <!-- Legend -->
    <div class="hotkey-legend">
      <b>Hotkeys:</b> <span class="kbd">0-9</span> Ubah Angka &middot; 
      <span class="kbd">Enter</span>/<span class="kbd">A</span> Setujui &middot; 
      <span class="kbd">K</span>/<span class="kbd">Spasi</span> Kosong &middot; 
      <span class="kbd">C</span> Centang &middot; 
      <span class="kbd">R</span> Ragu &middot; 
      <span class="kbd">&larr;</span>/<span class="kbd">&rarr;</span> Navigasi &middot; 
      <span class="kbd">Ctrl+Z</span> Undo
    </div>
  </div>
</main>

<button class="matrix-fab" onclick="toggleDrawer()">
  <span>&#9638;</span> Buka Matriks 721
</button>

<div class="drawer-modal" id="drawerModal">
  <div class="drawer-card">
    <div class="drawer-header">
      <h3>Matriks Antrean 721 Citra Prioritas</h3>
      <button class="nav-btn" onclick="toggleDrawer()">Tutup [Esc]</button>
    </div>
    <div class="matrix-grid" id="matrixGrid"></div>
  </div>
</div>

<script>
let items = [];
let currentIndex = 0;
let invertMode = false;

async function loadQueue() {
  try {
    const res = await fetch('/api/queue');
    items = await res.json();
    document.getElementById('statTotal').innerText = items.length;
    document.getElementById('jumpTotal').innerText = '/ ' + items.length;
    document.getElementById('jumpIdx').max = items.length;

    let firstUnreviewed = items.findIndex(it => it.status === 'unreviewed');
    currentIndex = firstUnreviewed >= 0 ? firstUnreviewed : 0;

    renderCurrent();
    renderStats();
    buildMatrix();
  } catch (e) {
    alert('Gagal memuat antrean: ' + e);
  }
}

function renderCurrent() {
  if (items.length === 0) return;
  const it = items[currentIndex];

  document.getElementById('jumpIdx').value = currentIndex + 1;
  document.getElementById('mainImg').src = it.image_b64;

  const cardTier = document.getElementById('cardTier');
  cardTier.innerText = it.alasan;
  cardTier.className = 'tier-pill ' + (it.tier === 1 ? 'tier-1' : 'tier-2');

  const cardStatus = document.getElementById('cardStatus');
  if (it.status === 'approved') {
    cardStatus.innerText = 'Disetujui';
    cardStatus.className = 'status-pill st-approved';
  } else if (it.status === 'corrected') {
    cardStatus.innerText = 'Dikoreksi';
    cardStatus.className = 'status-pill st-corrected';
  } else {
    cardStatus.innerText = 'Belum Diperiksa';
    cardStatus.className = 'status-pill st-unreviewed';
  }

  document.getElementById('currLblText').innerHTML = 'Label: <b>' + it.current_label + '</b>';
  document.getElementById('metaDetails').innerText = `${it.id_booklet} | ${it.halaman} | ${it.field_id} | ${it.box_id}`;
  document.getElementById('confReason').innerText = `Konfidensi AI: ${it.confidence.toFixed(2)} | Asli: ${it.orig_label}`;

  const twSec = document.getElementById('triwulanSection');
  if (it.field_id && it.field_id.toLowerCase().includes('triwulan')) {
    twSec.style.display = 'block';
  } else {
    twSec.style.display = 'none';
  }
}

function applyLabel(newLabel) {
  if (items.length === 0) return;
  const targetIdx = currentIndex;
  const it = items[targetIdx];

  // 1. Pembaruan instan lokal (Optimistic UI)
  it.current_label = newLabel;
  it.status = 'corrected';
  updateMatrixCard(targetIdx);

  // 2. LANGSUNG LOMPAT KE CROP BERIKUTNYA SECARA INSTAN (0ms delay, tanpa tunggu tombol lagi)
  nextItem();

  // 3. Kirim sinkronisasi ke server di background tanpa menghambat alur kerja
  fetch('/api/correct', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      nama_file_unik: it.nama_file_unik,
      new_label: newLabel,
      is_approval: false
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.success && data.stats) {
      updateStatsUI(data.stats);
    }
  })
  .catch(e => console.error('Gagal simpan:', e));
}

function approveCurrent() {
  if (items.length === 0) return;
  const targetIdx = currentIndex;
  const it = items[targetIdx];

  // 1. Pembaruan instan lokal
  it.status = 'approved';
  updateMatrixCard(targetIdx);

  // 2. LANGSUNG LOMPAT KE CROP BERIKUTNYA SECARA INSTAN (0ms delay)
  nextItem();

  // 3. Kirim sinkronisasi ke server di background
  fetch('/api/correct', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      nama_file_unik: it.nama_file_unik,
      new_label: it.current_label,
      is_approval: true
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.success && data.stats) {
      updateStatsUI(data.stats);
    }
  })
  .catch(e => console.error('Gagal simpan:', e));
}

async function undoLast() {
  try {
    const res = await fetch('/api/undo', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      const idx = items.findIndex(it => it.nama_file_unik === data.nama_file_unik);
      if (idx >= 0) {
        items[idx].current_label = data.current_label;
        items[idx].status = data.status;
        updateMatrixCard(idx);
        currentIndex = idx;
        renderCurrent();
      }
      updateStatsUI(data.stats);
    } else {
      alert(data.error || 'Gagal undo');
    }
  } catch (e) {
    console.error(e);
  }
}

function nextItem() {
  if (currentIndex < items.length - 1) {
    currentIndex++;
    renderCurrent();
  }
}

function prevItem() {
  if (currentIndex > 0) {
    currentIndex--;
    renderCurrent();
  }
}

function jumpTo(idx) {
  idx = parseInt(idx);
  if (idx >= 0 && idx < items.length) {
    currentIndex = idx;
    renderCurrent();
  }
}

function toggleInvert() {
  invertMode = !invertMode;
  const frame = document.getElementById('imgFrame');
  if (invertMode) frame.classList.add('invert-mode');
  else frame.classList.remove('invert-mode');
}

function toggleTheme() {
  const html = document.documentElement;
  const cur = html.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  document.getElementById('themeIcon').innerHTML = next === 'dark' ? '&#9790;' : '&#9788;';
  document.getElementById('themeLabel').innerText = next === 'dark' ? 'Dark' : 'Light';
}

function updateStatsUI(s) {
  document.getElementById('statReviewed').innerText = s.reviewed;
  document.getElementById('statApprove').innerText = s.approved;
  document.getElementById('statCorrect').innerText = s.corrected;
  document.getElementById('statPct').innerText = `(${s.progress_pct}%)`;
  document.getElementById('progFill').style.width = s.progress_pct + '%';

  const saveChip = document.getElementById('autoSaveChip');
  const saveIcon = document.getElementById('saveIcon');
  const saveStatus = document.getElementById('saveStatus');
  if (saveChip && s.last_saved) {
    if (s.dirty) {
      saveIcon.innerText = '⏳';
      saveStatus.innerText = 'Menyimpan...';
      saveChip.style.color = 'var(--c-seashell)';
      saveChip.style.borderColor = 'var(--c-seashell)';
    } else {
      saveIcon.innerText = '✓';
      saveStatus.innerText = 'Auto-Save ' + s.last_saved;
      saveChip.style.color = 'var(--c-teal)';
      saveChip.style.borderColor = 'var(--c-teal)';
    }
  }
}

async function triggerManualSync() {
  const saveIcon = document.getElementById('saveIcon');
  const saveStatus = document.getElementById('saveStatus');
  saveIcon.innerText = '⏳';
  saveStatus.innerText = 'Menyinkronkan...';
  const changedItems = items.filter(it => it.status !== 'unreviewed');
  try {
    const res = await fetch('/api/sync_batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(changedItems)
    });
    const d = await res.json();
    if (d.success) {
      saveIcon.innerText = '✓';
      saveStatus.innerText = `Tersimpan (${d.synced_count} sel)`;
      alert(`Berhasil! ${d.synced_count} hasil koreksi telah tersimpan permanen ke master CSV.`);
      renderStats();
    }
  } catch (e) {
    saveIcon.innerText = '⚠️';
    saveStatus.innerText = 'Gagal';
    alert('Gagal menyinkronkan: ' + e);
  }
}

async function renderStats() {
  try {
    const res = await fetch('/api/stats');
    const s = await res.json();
    updateStatsUI(s);
  } catch (e) {}
}

function buildMatrix() {
  const grid = document.getElementById('matrixGrid');
  grid.innerHTML = '';
  items.forEach((it, idx) => {
    const c = document.createElement('div');
    c.id = 'mat_' + idx;
    let cls = 'mat-card';
    if (it.status === 'approved') cls += ' c-approved';
    else if (it.status === 'corrected') cls += ' c-corrected';
    else if (it.tier === 1) cls += ' c-ragu';
    c.className = cls;
    c.onclick = () => {
      currentIndex = idx;
      renderCurrent();
      toggleDrawer();
    };
    c.innerHTML = `
      <img src="${it.image_b64}" loading="lazy">
      <div class="mat-label" id="mat_lbl_${idx}">${it.current_label}</div>
    `;
    grid.appendChild(c);
  });
}

function updateMatrixCard(idx) {
  const it = items[idx];
  const c = document.getElementById('mat_' + idx);
  if (!c) return;
  let cls = 'mat-card';
  if (it.status === 'approved') cls += ' c-approved';
  else if (it.status === 'corrected') cls += ' c-corrected';
  c.className = cls;
  const lbl = document.getElementById('mat_lbl_' + idx);
  if (lbl) lbl.innerText = it.current_label;
}

function toggleDrawer() {
  const d = document.getElementById('drawerModal');
  d.classList.toggle('open');
}

// Global Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
  if (document.activeElement.tagName === 'INPUT') return;

  if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
    e.preventDefault();
    undoLast();
    return;
  }

  if (e.key >= '0' && e.key <= '9') {
    e.preventDefault();
    const it = items[currentIndex];
    if (it && it.field_id && it.field_id.toLowerCase().includes('triwulan')) {
      if (e.key === '1') { applyLabel('I'); return; }
      if (e.key === '2') { applyLabel('II'); return; }
      if (e.key === '3') { applyLabel('III'); return; }
      if (e.key === '4') { applyLabel('IV'); return; }
    }
    applyLabel(e.key);
    return;
  }

  if (e.key === 'k' || e.key === 'K' || e.key === ' ') {
    e.preventDefault();
    applyLabel('kosong');
    return;
  }

  if (e.key === 'c' || e.key === 'C') {
    e.preventDefault();
    applyLabel('centang');
    return;
  }

  if (e.key === 'r' || e.key === 'R') {
    e.preventDefault();
    applyLabel('ragu_ragu');
    return;
  }

  if (e.key === 'Enter' || e.key === 'a' || e.key === 'A') {
    e.preventDefault();
    approveCurrent();
    return;
  }

  if (e.key === 'ArrowRight') {
    e.preventDefault();
    nextItem();
    return;
  }
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    prevItem();
    return;
  }

  if (e.key === 'i' || e.key === 'I') {
    e.preventDefault();
    toggleInvert();
    return;
  }

  if (e.key === 'Escape') {
    const d = document.getElementById('drawerModal');
    if (d.classList.contains('open')) d.classList.remove('open');
  }
});

loadQueue();
</script>
</body>
</html>
"""

class HITLHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def _send_response_data(self, status_code, content_type, body_bytes):
        self.close_connection = True
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body_bytes)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body_bytes)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_response_data(200, "text/html; charset=utf-8", HTML_PAGE.encode("utf-8"))
            return

        elif path == "/anomali" or path == "/anomali.html":
            if os.path.exists("anomali_viewer.html"):
                with open("anomali_viewer.html", "rb") as f:
                    self._send_response_data(200, "text/html; charset=utf-8", f.read())
                return
            else:
                self._send_response_data(404, "text/plain", b"anomali_viewer.html Not Found")
                return

        elif path == "/grid" or path == "/koreksi" or path == "/koreksi_massal" or path == "/koreksi_grid.html":
            if os.path.exists("koreksi_grid.html"):
                with open("koreksi_grid.html", "rb") as f:
                    self._send_response_data(200, "text/html; charset=utf-8", f.read())
                return
            else:
                self._send_response_data(404, "text/plain", b"koreksi_grid.html Not Found")
                return

        elif path == "/api/grid_items":
            query = parse_qs(parsed.query)
            label = query.get("label", ["1"])[0]
            try: page = int(query.get("page", ["1"])[0])
            except Exception: page = 1
            try: limit = int(query.get("limit", ["100"])[0])
            except Exception: limit = 100
            conf_filter = query.get("conf_filter", ["all"])[0]
            booklet = query.get("booklet", ["all"])[0]
            halaman = query.get("halaman", ["all"])[0]
            sort = query.get("sort", ["conf_asc"])[0]
            search = query.get("search", [""])[0]

            res = DATA_MGR.get_grid_items(
                label=label, page=page, limit=limit,
                conf_filter=conf_filter, booklet=booklet,
                halaman=halaman, sort=sort, search=search
            )
            self._send_response_data(200, "application/json", json.dumps(res).encode("utf-8"))
            return

        elif path == "/api/anomali":
            anom_csv = "hasil_labeling/anomalous_crops_report.csv"
            if os.path.exists(anom_csv):
                df_anom = pd.read_csv(anom_csv)
                data = df_anom.to_dict("records")
                self._send_response_data(200, "application/json", json.dumps(data).encode("utf-8"))
                return
            else:
                self._send_response_data(200, "application/json", b"[]")
                return

        elif path == "/api/image":
            query = parse_qs(parsed.query)
            fn = query.get("fn", [""])[0]
            lbl = query.get("lbl", [""])[0]
            img_path = os.path.join(BASE_IMAGE_DIR, lbl, fn)
            if not os.path.exists(img_path):
                for k in ["kosong", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "centang", "II", "III", "I", "ragu_ragu"]:
                    cand = os.path.join(BASE_IMAGE_DIR, k, fn)
                    if os.path.exists(cand):
                        img_path = cand
                        break
            if os.path.exists(img_path):
                with open(img_path, "rb") as img_f:
                    self._send_response_data(200, "image/png", img_f.read())
                return
            else:
                self._send_response_data(404, "text/plain", b"Image Not Found")
                return

        elif path == "/api/queue":
            self._send_response_data(200, "application/json", json.dumps(DATA_MGR.items).encode("utf-8"))
            return

        elif path == "/api/stats":
            self._send_response_data(200, "application/json", json.dumps(DATA_MGR.get_stats()).encode("utf-8"))
            return

        else:
            self._send_response_data(404, "text/plain", b"Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)

        if path == "/api/correct":
            try:
                payload = json.loads(post_body.decode("utf-8"))
                fn = payload["nama_file_unik"]
                new_lbl = str(payload["new_label"])
                is_approval = bool(payload.get("is_approval", False))
                op = payload.get("operator", "Operator BPS")

                res = DATA_MGR.save_action(fn, new_lbl, is_approval=is_approval, operator=op)
                self._send_response_data(200, "application/json", json.dumps(res).encode("utf-8"))
                return
            except Exception as e:
                err_payload = {"success": False, "error": str(e)}
                self._send_response_data(400, "application/json", json.dumps(err_payload).encode("utf-8"))
                return

        elif path == "/api/undo":
            res = DATA_MGR.undo_action()
            self._send_response_data(200, "application/json", json.dumps(res).encode("utf-8"))
            return

        elif path == "/api/bulk_correct":
            try:
                payload = json.loads(post_body.decode("utf-8"))
                items = payload if isinstance(payload, list) else payload.get("items", [])
                operator = payload.get("operator", "Operator BPS") if isinstance(payload, dict) else "Operator BPS"
                res = DATA_MGR.bulk_correct(items, operator=operator)
                self._send_response_data(200, "application/json", json.dumps(res).encode("utf-8"))
                return
            except Exception as e:
                err_payload = {"success": False, "error": str(e)}
                self._send_response_data(400, "application/json", json.dumps(err_payload).encode("utf-8"))
                return

        elif path == "/api/sync_batch":
            try:
                payload = json.loads(post_body.decode("utf-8"))
                items_list = payload if isinstance(payload, list) else payload.get("items", [])
                synced_count = 0
                for it in items_list:
                    fn = it.get("nama_file_unik")
                    new_lbl = str(it.get("current_label") or it.get("new_label"))
                    st = it.get("status", "corrected")
                    if fn and new_lbl and st != "unreviewed":
                        DATA_MGR.save_action(fn, new_lbl, is_approval=(st == "approved"), operator="Operator BPS (Batch Sync)")
                        synced_count += 1
                res = {"success": True, "synced_count": synced_count, "stats": DATA_MGR.get_stats()}
                self._send_response_data(200, "application/json", json.dumps(res).encode("utf-8"))
                return
            except Exception as e:
                err = {"success": False, "error": str(e)}
                self._send_response_data(400, "application/json", json.dumps(err).encode("utf-8"))
                return

        else:
            self._send_response_data(404, "text/plain", b"Not Found")

    def log_message(self, format, *args):
        pass


def main():
    port = PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    server_address = ("0.0.0.0", port)
    httpd = ThreadedTCPServer(server_address, HITLHandler)

    import socket
    lan_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    stats = DATA_MGR.get_stats()
    print("=" * 70)
    print("  BPS SERUTI - HITL LABELING VERIFIER (SEASHELL GARNET AFTERNOON)")
    print("=" * 70)
    print(f"Server berjalan pada:")
    print(f"  - Laptop (Lokal)         : http://localhost:{port}")
    print(f"  - Dari HP Hotspot (Wi-Fi): http://{lan_ip}:{port}")
    print(f"Total Antrean Prioritas : {stats['total']} citra")
    print(f"  - Tier 1 (Ambigu Fisik): {stats['tier1_total']} citra")
    print(f"  - Tier 2 (Conf <= 0.85): {stats['tier2_total']} citra")
    print(f"Status Terkini          : {stats['reviewed']} selesai ({stats['progress_pct']}%)")
    print("Tekan Ctrl+C untuk menghentikan server.")
    print("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server dihentikan oleh pengguna.")
        httpd.server_close()

if __name__ == "__main__":
    main()
