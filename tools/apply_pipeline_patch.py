import shutil
from pathlib import Path

html_file = Path("static/index.html")
backup_file = Path("static/index.html.bak_pipeline")

# 1. Buat backup jika belum ada
if not backup_file.exists():
    shutil.copy2(html_file, backup_file)
    print("Backup created at", backup_file)

html = html_file.read_text(encoding="utf-8")

# -------------------------------------------------------------------------
# A. Sisipkan stage-gate-modal sebelum modal DSRT
# -------------------------------------------------------------------------
modal_anchor = "<!-- =======================================================================\n       DSRT (Daftar Sampel Rumah Tangga) Master Prelist Modal"

stage_gate_modal_html = """  <!-- =======================================================================
       STAGE-GATE AUDIT (GATE 0-4) MODAL
       ======================================================================= -->
  <div id="stage-gate-modal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); backdrop-filter:blur(4px); z-index:9999; align-items:center; justify-content:center; padding:16px;">
    <div class="card" style="max-width:760px; width:100%; max-height:90vh; overflow-y:auto; padding:24px; background:var(--surface-1); border:1px solid var(--border); border-radius:var(--radius-md); box-shadow:0 20px 40px rgba(0,0,0,0.6); display:flex; flex-direction:column; gap:16px;">
      
      <!-- Modal Header -->
      <div style="display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--border); padding-bottom:12px;">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:34px; height:34px; border-radius:8px; background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.25); display:flex; align-items:center; justify-content:center; color:#38bdf8;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
          </div>
          <div>
            <div style="font-weight:700; font-size:15px; color:var(--text-primary); display:flex; align-items:center; gap:8px;">
              <span>Audit Gerbang Mutu (Stage-Gate 0–4)</span>
              <span id="sg-modal-booklet-badge" class="badge font-mono" style="font-size:11px;">-</span>
            </div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:2px;">
              Standar 2.073 Kotak Seruti Inti BPS · Validasi Presisi Geometris &amp; Kelayakan Scan
            </div>
          </div>
        </div>
        <button class="btn-ghost" onclick="closeStageGateModal()" style="padding:4px 8px; color:var(--text-secondary); cursor:pointer;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>

      <!-- Modal Body (Dynamic Container) -->
      <div id="sg-modal-body" style="display:flex; flex-direction:column; gap:14px;">
        <div style="padding:32px; text-align:center; color:var(--text-muted); font-size:13px;">
          Memuat data audit gerbang mutu...
        </div>
      </div>

      <!-- Modal Footer -->
      <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid var(--border); padding-top:14px;">
        <div style="font-size:11.5px; color:var(--text-muted);">
          Memeriksa 2.065 comb-box + 1 triwulan + 7 nomor halaman
        </div>
        <div style="display:flex; gap:10px;">
          <button class="btn btn-secondary btn-sm" id="btn-re-audit-booklet" onclick="reAuditCurrentBooklet()" style="display:flex; align-items:center; gap:6px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
            Audit Ulang Buklet
          </button>
          <button class="btn btn-secondary btn-sm" onclick="closeStageGateModal()">
            Tutup
          </button>
        </div>
      </div>

    </div>
  </div>

"""

if 'id="stage-gate-modal"' not in html:
    if modal_anchor in html:
        html = html.replace(modal_anchor, stage_gate_modal_html + modal_anchor)
        print("Stage-Gate modal inserted.")
    else:
        print("ERROR: modal_anchor not found!")

# -------------------------------------------------------------------------
# B. Perbarui startRealPipeline agar membaca selectedPipelineStages & inferenceModel
# -------------------------------------------------------------------------
old_start_real = """    async function startRealPipeline(targetBookletId = null) {
      if (AppState.batchRunning) {
        showToast('Pipeline sudah berjalan.', 'info');
        return;
      }

      const reqBody = {
        booklet_ids: targetBookletId ? [targetBookletId] : null,
        stages: ['crop', 'ocr', 'validate']
      };"""

new_start_real = """    async function startRealPipeline(targetBookletId = null) {
      if (AppState.batchRunning) {
        showToast('Pipeline sudah berjalan.', 'info');
        return;
      }

      const stages = (AppState.selectedPipelineStages && AppState.selectedPipelineStages.length > 0)
        ? AppState.selectedPipelineStages
        : ['gate', 'crop', 'ocr', 'validate', 'sync_db', 'auto_accept'];

      const reqBody = {
        booklet_ids: targetBookletId ? [targetBookletId] : null,
        stages: stages,
        model: AppState.inferenceModel || 'gemini_38_flash'
      };"""

if old_start_real in html:
    html = html.replace(old_start_real, new_start_real)
    print("startRealPipeline updated.")
else:
    print("WARNING: old_start_real pattern not found exact.")

# -------------------------------------------------------------------------
# C. Perbarui Toolbar & Tombol di renderViewPipeline
# -------------------------------------------------------------------------
old_top_btns = """            <button class="btn btn-secondary btn-sm" onclick="switchView('hitl')" style="display:flex; align-items:center; gap:6px;">"""
new_top_btns = """            <button class="btn btn-secondary btn-sm" onclick="runBatchStageGate()" style="display:flex; align-items:center; gap:6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                Audit Stage-Gate (51)
              </button>
            <button class="btn btn-secondary btn-sm" onclick="switchView('hitl')" style="display:flex; align-items:center; gap:6px;">"""

if old_top_btns in html:
    html = html.replace(old_top_btns, new_top_btns, 1)
    print("Audit Stage-Gate button added to top toolbar.")
else:
    print("WARNING: old_top_btns not found.")

# -------------------------------------------------------------------------
# D. Perbarui Card Alur Pipeline Otomasi dengan Selector Modular 7 Tahap
# -------------------------------------------------------------------------
old_track_block = """        <!-- Alur Pipeline Sederhana 3 Tahap Utama -->
        <div class="card" style="margin-bottom:24px; padding:22px 24px;">
          <div class="section-bar" style="margin-bottom:16px;">
            <div>
              <div class="section-heading" style="font-size:15px; font-weight:600;">Alur Pipeline Otomasi (3 Tahap Utama)</div>
              <div style="font-size:12px; color:var(--text-secondary); margin-top:2px;">
                Proses terintegrasi dari pindaian mentah hingga kuesioner digital terverifikasi.
              </div>
            </div>
            <span id="pipeline-stage-text" class="font-mono badge" style="padding:4px 10px; font-size:11.5px; background:var(--surface-2); border:1px solid var(--border);">
              ${currentStage}
            </span>
          </div>

          <!-- Visual 3-Stage Track -->
          <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:14px; margin-bottom:18px;">
            <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:14px 16px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Tahap 1</span>
                <span class="badge" style="font-size:10.5px; background:rgba(16,185,129,0.1); color:#10b981; border:1px solid rgba(16,185,129,0.25);">Bypass 78.3%</span>
              </div>
              <div style="font-weight:600; font-size:13.5px; margin:4px 0;">Ekstraksi &amp; Filter Kosong</div>
              <div style="font-size:12px; color:var(--text-secondary); line-height:1.45;">Deskewing, penjajaran grid 7-tier, &amp; deteksi kotak kosong OpenCV (langsung '_' tanpa biaya model).</div>
            </div>

            <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:14px 16px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Tahap 2</span>
                <span class="badge" style="font-size:10.5px; background:rgba(56,189,248,0.1); color:#38bdf8; border:1px solid rgba(56,189,248,0.25);">AI Vision</span>
              </div>
              <div style="font-weight:600; font-size:13.5px; margin:4px 0;">Inferensi AI &amp; OCR</div>
              <div style="font-size:12px; color:var(--text-secondary); line-height:1.45;">Pembacaan otomatis isi formulir terisi (kisi 50-crop / model lokal) dengan ekstraksi confidence score.</div>
            </div>

            <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:14px 16px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Tahap 3</span>
                <span class="badge" style="font-size:10.5px; background:rgba(168,85,247,0.1); color:#a855f7; border:1px solid rgba(168,85,247,0.25);">21 Aturan BPS</span>
              </div>
              <div style="font-weight:600; font-size:13.5px; margin:4px 0;">Validasi Semantik &amp; Integrasi</div>
              <div style="font-size:12px; color:var(--text-secondary); line-height:1.45;">Pengecekan konsistensi silang, subtotal aritmetika vertikal, &amp; agregasi ke kuesioner digital.</div>
            </div>
          </div>"""

new_track_block = """        <!-- Alur Pipeline Otomasi Terintegrasi Modular (7 Tahap) -->
        <div class="card" style="margin-bottom:24px; padding:22px 24px;">
          <div class="section-bar" style="margin-bottom:16px;">
            <div>
              <div class="section-heading" style="font-size:15px; font-weight:600;">Alur Pipeline Otomasi Terintegrasi</div>
              <div style="font-size:12px; color:var(--text-secondary); margin-top:2px;">
                Pilih modul tahapan yang ingin dijalankan secara granular atau otomatis penuh.
              </div>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span id="pipeline-stage-text" class="font-mono badge" style="padding:4px 10px; font-size:11.5px; background:var(--surface-2); border:1px solid var(--border);">
                ${currentStage}
              </span>
            </div>
          </div>

          <!-- Modular Stage Selector Pills -->
          <div style="margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
              <span style="font-size:12px; font-weight:600; color:var(--text-secondary);">Tahapan Eksekusi Aktif:</span>
              <div style="display:flex; gap:8px;">
                <button type="button" class="btn-ghost" onclick="selectAllPipelineStages(true)" style="font-size:11.5px; padding:2px 8px;">Pilih Semua (7)</button>
                <button type="button" class="btn-ghost" onclick="selectAllPipelineStages(false)" style="font-size:11.5px; padding:2px 8px;">Standar Produksi (6)</button>
              </div>
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
              ${PIPELINE_AVAILABLE_STAGES.map(s => {
                const isAct = activeStages.includes(s.id);
                return `
                  <button type="button" class="pill-tab ${isAct ? 'active' : ''}" onclick="togglePipelineStage('${s.id}')"
                          title="${s.desc}"
                          style="display:flex; align-items:center; gap:6px; padding:6px 12px; font-size:12px; font-weight:600; cursor:pointer; border:1px solid ${isAct ? 'var(--accent)' : 'var(--border)'}; background:${isAct ? 'rgba(56,189,248,0.1)' : 'var(--surface-2)'}; color:${isAct ? 'var(--accent)' : 'var(--text-secondary)'};">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${isAct ? '2.5' : '1.75'}">
                      ${isAct ? '<polyline points="20 6 9 17 4 12"/>' : '<circle cx="12" cy="12" r="9"/>'}
                    </svg>
                    <span>${s.name}</span>
                  </button>
                `;
              }).join('')}
            </div>
          </div>"""

if old_track_block in html:
    html = html.replace(old_track_block, new_track_block)
    print("Pipeline stage selector cards replaced.")
else:
    print("WARNING: old_track_block not found.")

# -------------------------------------------------------------------------
# E. Tambah Kolom 'Gerbang Mutu' di pipeline-jobs-table
# -------------------------------------------------------------------------
old_thead = """                <thead>
                  <tr>
                    <th>ID Buklet</th>
                    <th>Nama KRT</th>
                    <th>Tahap Saat Ini</th>
                    <th style="width:160px;">Progres</th>
                    <th>Kotak Kosong (OpenCV)</th>
                    <th>Kotak Terisi (AI/OCR)</th>
                    <th>Status</th>
                    <th style="text-align:right;">Aksi</th>
                  </tr>
                </thead>"""

new_thead = """                <thead>
                  <tr>
                    <th>ID Buklet</th>
                    <th>Nama KRT</th>
                    <th>Gerbang Mutu</th>
                    <th>Tahap Saat Ini</th>
                    <th style="width:140px;">Progres</th>
                    <th>Kotak Kosong (OpenCV)</th>
                    <th>Kotak Terisi (AI/OCR)</th>
                    <th>Status</th>
                    <th style="text-align:right;">Aksi</th>
                  </tr>
                </thead>"""

if old_thead in html:
    html = html.replace(old_thead, new_thead)
    print("Table header updated with Gerbang Mutu.")
else:
    print("WARNING: old_thead not found.")

old_tbody_row = """                      <td class="font-mono" style="font-weight:600;">${j.id}</td>
                      <td>${j.krt || '-'}</td>
                      <td style="font-size:12px; color:var(--text-secondary);">${j.stage || 'Selesai'}</td>"""

new_tbody_row = """                      <td class="font-mono" style="font-weight:600;">${j.id}</td>
                      <td>${j.krt || '-'}</td>
                      <td>${renderGateBadge(j.id, j.gate_status)}</td>
                      <td style="font-size:12px; color:var(--text-secondary);">${j.stage || 'Selesai'}</td>"""

if old_tbody_row in html:
    html = html.replace(old_tbody_row, new_tbody_row)
    print("Table row updated with renderGateBadge.")
else:
    print("WARNING: old_tbody_row not found.")

# Update empty row colspan
old_empty = '<td colspan="8" style="text-align:center; padding:32px; color:var(--text-muted);">'
new_empty = '<td colspan="9" style="text-align:center; padding:32px; color:var(--text-muted);">'
if old_empty in html:
    html = html.replace(old_empty, new_empty)
    print("Empty row colspan updated to 9.")

# -------------------------------------------------------------------------
# F. Tambahkan definisi PIPELINE_AVAILABLE_STAGES & JS functions
# -------------------------------------------------------------------------
js_functions = """
    // =========================================================================
    // MODULAR PIPELINE STAGES & STAGE-GATE MODAL HELPERS
    // =========================================================================
    const PIPELINE_AVAILABLE_STAGES = [
      { id: 'gate', name: 'Gate 0-4', label: 'Gerbang Mutu (0-4)', desc: 'Validasi kelayakan scan 2.073 kotak' },
      { id: 'crop', name: 'Crop & Filter', label: 'Grid Cropping', desc: 'Ekstraksi kotak & filter kosong OpenCV (78.3%)' },
      { id: 'ocr', name: 'OCR AI', label: 'Vision OCR Multi-Tier', desc: 'Pembacaan cerdas model Gemini 3.8 Flash / CNN' },
      { id: 'validate', name: 'Validasi Semantik', label: '21 Relasi Semantik', desc: 'Cek konsistensi silang & subtotal vertikal BPS' },
      { id: 'sync_db', name: 'Sync DB', label: 'Sinkronisasi Database', desc: 'Simpan otomatis ke booklet_field_values & inferences' },
      { id: 'auto_accept', name: 'Auto-Accept', label: 'Smart Auto-Accept', desc: 'Lulus otomatis tanpa HITL (conf ≥ 80% & nol konsisten)' },
      { id: 'dsrt_sync', name: 'DSRT Sync', label: 'DSRT Prelist Sync', desc: 'Penyelarasan identitas KOP dengan sampel aktif' }
    ];

    function renderGateBadge(bookletId, gateStatus) {
      const gs = gateStatus || 'UNKNOWN';
      if (gs === 'PROCEED_TO_OCR' || gs === 'PASS') {
        return `<button class="badge badge-pass" onclick="openStageGateModal('${bookletId}')" style="cursor:pointer; border:none; display:inline-flex; align-items:center; gap:4px;" title="Lihat hasil audit Gate 0-4">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Gate PASS
        </button>`;
      } else if (gs === 'FLAG_FOR_HITL_REVIEW' || gs === 'FLAGGED') {
        return `<button class="badge badge-warning" onclick="openStageGateModal('${bookletId}')" style="cursor:pointer; border:none; display:inline-flex; align-items:center; gap:4px;" title="Lihat catatan audit Gate 0-4">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> Gate FLAGGED
        </button>`;
      } else if (gs === 'REJECT_INCOMPLETE' || gs === 'REJECT' || gs.startsWith('REJECT')) {
        return `<button class="badge badge-error" onclick="openStageGateModal('${bookletId}')" style="cursor:pointer; border:none; display:inline-flex; align-items:center; gap:4px;" title="Lihat rincian penolakan">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> Gate REJECT
        </button>`;
      } else {
        return `<button class="badge" onclick="runSingleStageGate('${bookletId}')" style="cursor:pointer; background:var(--surface-2); border:1px solid var(--border); color:var(--text-secondary); display:inline-flex; align-items:center; gap:4px;" title="Audit Gerbang Mutu">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg> Audit Gate
        </button>`;
      }
    }

    window.togglePipelineStage = function(stageId) {
      if (!AppState.selectedPipelineStages) {
        AppState.selectedPipelineStages = ['gate', 'crop', 'ocr', 'validate', 'sync_db', 'auto_accept'];
      }
      const idx = AppState.selectedPipelineStages.indexOf(stageId);
      if (idx > -1) {
        if (AppState.selectedPipelineStages.length <= 1) {
          showToast('Minimal satu tahap pipeline harus aktif.', 'warning');
          return;
        }
        AppState.selectedPipelineStages.splice(idx, 1);
      } else {
        AppState.selectedPipelineStages.push(stageId);
      }
      renderViewPipeline(document.getElementById('main-viewport'));
    };

    window.selectAllPipelineStages = function(enableAll = true) {
      if (enableAll) {
        AppState.selectedPipelineStages = PIPELINE_AVAILABLE_STAGES.map(s => s.id);
      } else {
        AppState.selectedPipelineStages = ['gate', 'crop', 'ocr', 'validate', 'sync_db', 'auto_accept'];
      }
      renderViewPipeline(document.getElementById('main-viewport'));
    };

    window.runBatchStageGate = async function() {
      try {
        showToast('Menjalankan audit Gerbang Mutu (Stage-Gate 0-4) untuk seluruh buklet...', 'info');
        const res = await fetch('/api/pipeline/stage-gate/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        const data = await res.json();
        if (res.ok) {
          showToast(data.message || 'Audit Stage-Gate 51 buklet selesai.', 'success');
          await loadRealPipelineStatus();
          renderViewPipeline(document.getElementById('main-viewport'));
        } else {
          showToast('Gagal audit stage-gate: ' + (data.message || 'Error'), 'error');
        }
      } catch (err) {
        showToast('Error koneksi audit stage-gate: ' + err.message, 'error');
      }
    };

    window.runSingleStageGate = async function(bookletId) {
      try {
        showToast(`Mengaudit Gerbang Mutu ${bookletId}...`, 'info');
        const res = await fetch('/api/pipeline/stage-gate/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ booklet_ids: [bookletId] })
        });
        const data = await res.json();
        if (res.ok) {
          showToast(`Audit ${bookletId} selesai`, 'success');
          await loadRealPipelineStatus();
          renderViewPipeline(document.getElementById('main-viewport'));
          openStageGateModal(bookletId);
        } else {
          showToast('Gagal audit: ' + (data.message || 'Error'), 'error');
        }
      } catch (err) {
        showToast('Error koneksi audit: ' + err.message, 'error');
      }
    };

    window.openStageGateModal = async function(bookletId) {
      const modal = document.getElementById('stage-gate-modal');
      if (!modal) return;
      modal.style.display = 'flex';

      const bBadge = document.getElementById('sg-modal-booklet-badge');
      if (bBadge) bBadge.textContent = bookletId;
      AppState.currentAuditingBooklet = bookletId;

      const bodyEl = document.getElementById('sg-modal-body');
      if (bodyEl) {
        bodyEl.innerHTML = `
          <div style="padding:40px; text-align:center; color:var(--text-muted); font-size:13px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin-slow" style="margin-bottom:8px;"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 10 10"/></svg>
            <div>Mengambil catatan audit gerbang mutu untuk ${bookletId}...</div>
          </div>
        `;
      }

      try {
        const res = await fetch(`/api/pipeline/stage-gate/${bookletId}`);
        if (!res.ok) {
          if (bodyEl) {
            bodyEl.innerHTML = `
              <div style="padding:24px; text-align:center; color:var(--text-muted); background:var(--surface-2); border-radius:6px; border:1px solid var(--border);">
                <div style="color:var(--text-primary); font-weight:600; margin-bottom:6px;">Belum Ada Data Audit Stage-Gate</div>
                <div style="font-size:12px; margin-bottom:14px; color:var(--text-secondary);">Buklet ${bookletId} belum melalui evaluasi gerbang mutu presisi 2.073 kotak.</div>
                <button class="btn btn-primary btn-sm" onclick="runSingleStageGate('${bookletId}')">Jalankan Audit Gerbang Mutu Sekarang</button>
              </div>
            `;
          }
          return;
        }
        const data = await res.json();
        renderStageGateModalContent(data);
      } catch (err) {
        if (bodyEl) {
          bodyEl.innerHTML = `<div style="color:#ef4444; padding:16px;">Gagal memuat audit: ${err.message}</div>`;
        }
      }
    };

    window.closeStageGateModal = function() {
      const modal = document.getElementById('stage-gate-modal');
      if (modal) modal.style.display = 'none';
    };

    window.reAuditCurrentBooklet = async function() {
      const bId = AppState.currentAuditingBooklet;
      if (!bId) return;
      await runSingleStageGate(bId);
    };

    function renderStageGateModalContent(data) {
      const bodyEl = document.getElementById('sg-modal-body');
      if (!bodyEl) return;

      const dec = data.decision || 'UNKNOWN';
      let decColor = '#38bdf8';
      let decBg = 'rgba(56,189,248,0.1)';
      let decBorder = 'rgba(56,189,248,0.25)';
      let decLabel = 'PROCEED TO OCR (PASS)';
      let decIcon = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';

      if (dec === 'FLAG_FOR_HITL_REVIEW' || dec === 'FLAGGED') {
        decColor = '#f59e0b';
        decBg = 'rgba(245,158,11,0.1)';
        decBorder = 'rgba(245,158,11,0.25)';
        decLabel = 'FLAG FOR HITL REVIEW';
        decIcon = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
      } else if (dec.startsWith('REJECT')) {
        decColor = '#ef4444';
        decBg = 'rgba(239,68,68,0.1)';
        decBorder = 'rgba(239,68,68,0.25)';
        decLabel = 'REJECT INCOMPLETE';
        decIcon = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
      }

      const gates = data.gates || {};
      const g0 = gates.gate_0_iqg || {};
      const g1 = gates.gate_1_alignment || gates.gate_1_affine || {};
      const g2 = gates.gate_2_digit_count || gates.gate_2_counts || {};
      const g3 = gates.gate_3_snapping_quality || gates.gate_3_snapping || {};

      const reasons = data.reasons || [];

      bodyEl.innerHTML = `
        <!-- Decision Banner -->
        <div style="background:${decBg}; border:1px solid ${decBorder}; border-radius:var(--radius-sm); padding:14px 16px; display:flex; align-items:flex-start; gap:12px;">
          <div style="color:${decColor}; margin-top:2px;">${decIcon}</div>
          <div style="flex:1;">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
              <span style="font-weight:700; font-size:14px; color:${decColor};">${decLabel}</span>
              <span class="badge font-mono" style="font-size:11px; background:var(--surface-1);">${data.routing || 'ROUTED'}</span>
            </div>
            <div style="font-size:12px; color:var(--text-secondary); line-height:1.45;">
              ${reasons.length > 0 ? reasons.join(' · ') : 'Seluruh gerbang kelayakan geometris dan presisi 2.073 kotak terpenuhi dengan baik.'}
            </div>
          </div>
        </div>

        <!-- 4 Gerbang Mutu Breakdown -->
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:12px;">
          
          <!-- Gate 0: IQG -->
          <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:12px 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
              <span style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-muted);">Gate 0 · Input Quality Gate</span>
              <span class="badge" style="font-size:10px; background:${g0.status==='PASS'?'rgba(16,185,129,0.1)':'rgba(239,68,68,0.1)'}; color:${g0.status==='PASS'?'#10b981':'#ef4444'};">
                ${g0.status || 'PASS'}
              </span>
            </div>
            <div style="font-size:12px; color:var(--text-secondary); line-height:1.4;">
              Pemeriksaan kontras tinta, washed-out blank, serta noise margin roller scanner (7 halaman pindaian).
            </div>
          </div>

          <!-- Gate 1: Alignment & Affine -->
          <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:12px 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
              <span style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-muted);">Gate 1 · Penjajaran &amp; Affine</span>
              <span class="badge" style="font-size:10px; background:${g1.status==='PASS'?'rgba(16,185,129,0.1)':'rgba(245,158,11,0.1)'}; color:${g1.status==='PASS'?'#10b981':'#f59e0b'};">
                ${g1.status || 'PASS'}
              </span>
            </div>
            <div style="font-size:12px; color:var(--text-secondary); line-height:1.4;">
              Deviasi rasio aspek sudut &amp; matriks transformasi affine per halaman (toleransi &lt; 5.0%).
            </div>
          </div>

          <!-- Gate 2: Completeness 2.073 Kotak -->
          <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:12px 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
              <span style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-muted);">Gate 2 · Invariant 2.073 Kotak</span>
              <span class="badge" style="font-size:10px; background:${g2.status==='PASS'?'rgba(16,185,129,0.1)':'rgba(245,158,11,0.1)'}; color:${g2.status==='PASS'?'#10b981':'#f59e0b'};">
                ${g2.status || 'PASS'}
              </span>
            </div>
            <div style="font-size:12px; color:var(--text-secondary); line-height:1.4;">
              Kelengkapan Zero-Loss: 2.065 kotak digit comb-box + 1 kotak triwulan romawi + 7 nomor halaman.
            </div>
          </div>

          <!-- Gate 3: Substantive Snapping -->
          <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:12px 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
              <span style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-muted);">Gate 3 · Snapping Geometris</span>
              <span class="badge" style="font-size:10px; background:${g3.status==='PASS'?'rgba(16,185,129,0.1)':'rgba(245,158,11,0.1)'}; color:${g3.status==='PASS'?'#10b981':'#f59e0b'};">
                ${g3.status || 'PASS'}
              </span>
            </div>
            <div style="font-size:12px; color:var(--text-secondary); line-height:1.4;">
              Presisi penempelan kisi fisik kuesioner, integritas batas border, &amp; sanitasi overlap.
            </div>
          </div>

        </div>
      `;
    }
"""

# Let's find where to insert js_functions (before renderViewPipeline or after updatePipelineUIElements)
target_js_anchor = "    function renderViewPipeline(el) {"
if target_js_anchor in html:
    html = html.replace(target_js_anchor, js_functions + "\n" + target_js_anchor)
    print("JS helper functions and PIPELINE_AVAILABLE_STAGES inserted.")
else:
    print("ERROR: target_js_anchor not found!")

# Also ensure AppState.selectedPipelineStages has a default in renderViewPipeline
old_render_init = """      AppState.pipelineSubMode = AppState.pipelineSubMode || 'jobs';

      if (!AppState.pipelinePageSize) AppState.pipelinePageSize = 10;"""

new_render_init = """      AppState.pipelineSubMode = AppState.pipelineSubMode || 'jobs';
      if (!AppState.selectedPipelineStages) {
        AppState.selectedPipelineStages = ['gate', 'crop', 'ocr', 'validate', 'sync_db', 'auto_accept'];
      }
      const activeStages = AppState.selectedPipelineStages;

      if (!AppState.pipelinePageSize) AppState.pipelinePageSize = 10;"""

if old_render_init in html:
    html = html.replace(old_render_init, new_render_init)
    print("activeStages initialized in renderViewPipeline.")
else:
    print("WARNING: old_render_init not found.")

html_file.write_text(html, encoding="utf-8")
print("HTML successfully written! New length:", len(html))
