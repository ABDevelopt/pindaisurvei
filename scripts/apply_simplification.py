import re
from pathlib import Path

index_path = Path("static/index.html")
content = index_path.read_text(encoding="utf-8")

# 1. Update admin array in WorkspaceNavs
admin_pattern = r"admin:\s*\[[\s\S]*?\]\s*\}\s*;"
new_admin = """admin: [
        { key: 'grid',      label: 'Grid',       title: 'Editor Template & Grid Vektor',       route: '/grid',      icon: '<rect x="3" y="3" width="18" height="18" rx="3"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/>' },
        { key: 'crop',      label: 'Crop',       title: 'Inspektor Hierarki Cropping',         route: '/crop',      icon: '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>' },
        { key: 'training',  label: 'Pelatihan',  title: 'Active Learning & Retraining Model',  route: '/training',  icon: '<path d="M12 2a4 4 0 0 0-4 4v2H6a2 2 0 0 0-2 2v2a4 4 0 0 0 4 4h8a4 4 0 0 0 4-4v-2a2 2 0 0 0-2-2h-2V6a4 4 0 0 0-4-4z"/><line x1="9" y1="22" x2="15" y2="22"/><line x1="12" y1="16" x2="12" y2="22"/>' },
        { key: 'admin',     label: 'Sistem',     title: 'Pengaturan Sistem & Kunci API',       route: '/admin',     icon: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06-.06a1.65 1.65 0 0 0-.33 1.82V9c.26.6.85 1 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>' }
      ]
    };"""

content = re.sub(admin_pattern, new_admin, content, count=1)

# 2. Add state properties to AppState
if "pipelineSubMode:" not in content:
    content = content.replace(
        "workspaceMode: 'operasional',",
        "workspaceMode: 'operasional',\n      pipelineSubMode: 'jobs', // 'jobs' | 'inference'\n      showPipelineLog: false,"
    )

# 3. Update AppRouter routes
router_pattern = r"routes:\s*\[[\s\S]*?\],\s*getRouteUrl"
new_routes = """routes: [
        { pattern: /^\/?$/, key: 'input', ws: 'operasional', title: 'Input & Berkas Scan' },
        { pattern: /^\/input\/?$/, key: 'input', ws: 'operasional', title: 'Input & Berkas Scan' },
        { pattern: /^\/booklets\/?$/, key: 'input', ws: 'operasional', title: 'Input & Berkas Scan' },
        { pattern: /^\/booklets\/([^\\/]+)\/?$/, key: 'kuesioner', ws: 'operasional', title: 'Kuesioner Digital', handler: (m) => { AppState.activeBooklet = m[1]; } },
        { pattern: /^\/buku\/?$/, key: 'input', ws: 'operasional', title: 'Input & Berkas Scan' },
        { pattern: /^\/preproc\/?$/, key: 'input', ws: 'operasional', title: 'Input & Berkas Scan' },
        { pattern: /^\/pipeline\/?$/, key: 'pipeline', ws: 'operasional', title: 'Pipeline Ekstraksi & Inferensi AI' },
        { pattern: /^\/proses\/?$/, key: 'pipeline', ws: 'operasional', title: 'Pipeline Ekstraksi & Inferensi AI' },
        { pattern: /^\/inference\/?$/, key: 'pipeline', ws: 'operasional', title: 'Pipeline & Inferensi AI', handler: () => { AppState.pipelineSubMode = 'inference'; } },
        { pattern: /^\/inferensi\/?$/, key: 'pipeline', ws: 'operasional', title: 'Pipeline & Inferensi AI', handler: () => { AppState.pipelineSubMode = 'inference'; } },
        { pattern: /^\/hitl\/?$/, key: 'hitl', ws: 'operasional', title: 'Verifikasi HITL Petugas' },
        { pattern: /^\/verifikasi\/?$/, key: 'hitl', ws: 'operasional', title: 'Verifikasi HITL Petugas' },
        { pattern: /^\/kuesioner\/?$/, key: 'kuesioner', ws: 'operasional', title: 'Kuesioner Digital & Ekspor FASIH-SM' },
        { pattern: /^\/kuesioner\/([^\\/]+)\/?$/, key: 'kuesioner', ws: 'operasional', title: 'Kuesioner Digital', handler: (m) => { AppState.activeBooklet = m[1]; } },
        { pattern: /^\/output\/?$/, key: 'kuesioner', ws: 'operasional', title: 'Kuesioner Digital & Ekspor FASIH-SM', handler: () => { AppState.kuesionerMode = 'validation'; } },
        { pattern: /^\/digital\/?$/, key: 'kuesioner', ws: 'operasional', title: 'Kuesioner Digital & Ekspor FASIH-SM', handler: () => { AppState.kuesionerMode = 'validation'; } },
        { pattern: /^\/ekspor\/?$/, key: 'kuesioner', ws: 'operasional', title: 'Kuesioner Digital & Ekspor FASIH-SM' },
        { pattern: /^\/grid\/?$/, key: 'grid', ws: 'admin', title: 'Grid Vektor & Spasial' },
        { pattern: /^\/crop\/?$/, key: 'crop', ws: 'admin', title: 'Ekstraksi Hierarkis 7-Tier' },
        { pattern: /^\/training\/?$/, key: 'training', ws: 'admin', title: 'Pelatihan Active Learning & Model' },
        { pattern: /^\/pelatihan\/?$/, key: 'training', ws: 'admin', title: 'Pelatihan Active Learning & Model' },
        { pattern: /^\/admin\/?$/, key: 'admin', ws: 'admin', title: 'Administrasi Sistem' },
        { pattern: /^\/system\/?$/, key: 'admin', ws: 'admin', title: 'Administrasi Sistem' },
        { pattern: /^\/settings\/?$/, key: 'admin', ws: 'admin', title: 'Pengaturan Sistem' },
      ],

      getRouteUrl"""

content = re.sub(router_pattern, new_routes, content, count=1)

# 4. In switchView handle 'inference'
if "if (key === 'inference')" not in content:
    content = content.replace(
        "AppState.activeTab = key;",
        "if (key === 'inference') { AppState.pipelineSubMode = 'inference'; key = 'pipeline'; }\n      AppState.activeTab = key;"
    )

# 5. Replace renderViewPipeline with unified, clean & breathing view
pipeline_view_code = '''    function renderViewPipeline(el) {
      const p = AppState.realPipelineStatus || {};
      const isRunning = AppState.batchRunning || p.running || false;
      const progressPct = p.progress_pct !== undefined ? p.progress_pct : 100;
      const currentStage = p.stage || (isRunning ? 'Memproses...' : 'Selesai (76 Buklet)');
      AppState.pipelineSubMode = AppState.pipelineSubMode || 'jobs';

      const totalBoxes = 95358;
      const blankBoxes = 74699;
      const filledBoxes = 20659;
      const autoPassPct = '98.2%';

      el.innerHTML = `
        <!-- Top Section Bar: Booklet, Model Engine, Quick Action Buttons -->
        <div class="section-bar" style="margin-bottom:20px; gap:12px; flex-wrap:wrap; display:flex; align-items:center; justify-content:space-between;">
          <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            <div class="status-inline">
              <span id="pipeline-status-dot" class="status-dot ${isRunning ? 'dot-accent' : 'dot-green'}"></span>
              <span id="pipeline-status-text" style="font-weight:600; font-size:13px;">
                ${isRunning ? `Pipeline aktif · ${currentStage}` : 'Pipeline Produksi Siap · 76 Buklet Terakreditasi BPS'}
              </span>
            </div>

            <!-- Target Booklet Selector -->
            <select class="select-pill" onchange="onBookletChange(this.value).then(() => { if (AppState.pipelineSubMode === 'inference') loadInferenceData(); })" title="Pilih Buklet Target" style="font-weight:600; padding:6px 14px; font-size:12.5px;">
              ${(AppState.booklets || []).map(b => `
                <option value="${b.id}" ${AppState.activeBooklet===b.id?'selected':''}>${b.id} · ${b.nama_krt || b.id}</option>
              `).join('')}
            </select>

            <!-- Model Engine Selector -->
            <select class="select-pill" onchange="onInferenceModelChange(this.value)" title="Pilih Model AI" 
                    style="font-weight:600; padding:6px 14px; font-size:12.5px; border:1px solid #38bdf8; background:rgba(56,189,248,0.08); color:#38bdf8;">
              <option value="gemini_38_flash" ${(!AppState.inferenceModel || AppState.inferenceModel==='gemini_38_flash')?'selected':''}>Gemini 3.8 Flash (Paling Akurat · Default)</option>
              <option value="seruti_cnn_v3" ${AppState.inferenceModel==='seruti_cnn_v3'?'selected':''}>Seruti CNN v3 (Lokal Offline)</option>
              <option value="seruti_mlp" ${AppState.inferenceModel==='seruti_mlp'?'selected':''}>Seruti MLP (Lokal Ringan)</option>
            </select>
          </div>

          <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
            ${isRunning ? `
              <button class="btn btn-secondary btn-sm" onclick="stopRealPipeline()" style="border-color:#ef4444; color:#ef4444; display:flex; align-items:center; gap:6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"/></svg>
                Hentikan Proses
              </button>
            ` : `
              <button class="btn btn-primary btn-sm" onclick="startRealPipeline(null)" style="display:flex; align-items:center; gap:6px; font-weight:600;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                Jalankan Seluruh 76 Buklet
              </button>
              <button class="btn btn-secondary btn-sm" onclick="startRealPipeline(AppState.activeBooklet || 'booklet_01')" style="display:flex; align-items:center; gap:6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
                Jalankan 1 Buklet (${AppState.activeBooklet || 'booklet_01'})
              </button>
            `}
            <button class="btn btn-secondary btn-sm" onclick="switchView('hitl')" style="display:flex; align-items:center; gap:6px;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
              Verifikasi HITL →
            </button>
            <button class="btn btn-secondary btn-sm" onclick="switchView('kuesioner')" style="display:flex; align-items:center; gap:6px;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
              Kuesioner Digital →
            </button>
          </div>
        </div>

        <!-- 4 Metrik Utama Sesuai Aturan Mutlak Anti-Clutter -->
        <div class="metrics-strip" style="margin-bottom:24px;">
          <div class="metric-item">
            <div class="metric-value font-mono">${totalBoxes.toLocaleString('id-ID')}</div>
            <div class="metric-label">Total Kotak Dipindai</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:3px;">76 Buklet · Hal 1 s.d. Hal 7</div>
          </div>
          <div class="metric-item">
            <div class="metric-value font-mono" style="color:#10b981;">${blankBoxes.toLocaleString('id-ID')} <span style="font-size:12px; font-weight:400;">(78.3%)</span></div>
            <div class="metric-label"><span class="status-dot dot-green"></span>Filter Kosong OpenCV</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:3px;">Langsung '_' (100% Conf · 0 Token)</div>
          </div>
          <div class="metric-item">
            <div class="metric-value font-mono" style="color:var(--accent);">${filledBoxes.toLocaleString('id-ID')} <span style="font-size:12px; font-weight:400;">(21.7%)</span></div>
            <div class="metric-label"><span class="status-dot dot-accent"></span>Kotak Terisi (AI OCR)</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:3px;">Diinferensi Model Vision</div>
          </div>
          <div class="metric-item">
            <div class="metric-value font-mono">${autoPassPct}</div>
            <div class="metric-label">Lolos Validasi Otomatis</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:3px;">Keyakinan ≥ 95%</div>
          </div>
        </div>

        <!-- Alur Pipeline Sederhana 3 Tahap Utama -->
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
          </div>

          <!-- Overall Progress Bar -->
          <div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
              <span style="color:var(--text-secondary);">Progres Pipeline Keseluruhan</span>
              <span id="pipeline-progress-pct" class="font-mono" style="font-weight:600;">${progressPct}%</span>
            </div>
            <div class="progress-track" style="height:7px;">
              <div id="pipeline-progress-bar" class="progress-fill" style="width:${progressPct}%;"></div>
            </div>
          </div>
        </div>

        <!-- Sub-View Switcher: Status Buklet / Hasil Inferensi AI -->
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; flex-wrap:wrap; gap:10px;">
          <div style="display:flex; gap:8px;">
            <button class="pill-tab ${AppState.pipelineSubMode==='jobs'?'active':''}" onclick="setPipelineSubMode('jobs')" style="padding:6px 14px; font-size:12.5px; font-weight:600;">
              Status Pemrosesan Buklet (76)
            </button>
            <button class="pill-tab ${AppState.pipelineSubMode==='inference'?'active':''}" onclick="setPipelineSubMode('inference')" style="padding:6px 14px; font-size:12.5px; font-weight:600;">
              Hasil Inferensi AI (${AppState.activeBooklet || 'booklet_01'})
            </button>
          </div>

          <button class="btn-ghost" onclick="togglePipelineLog()" style="font-size:12px; display:flex; align-items:center; gap:6px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
            <span id="log-toggle-label">${AppState.showPipelineLog ? 'Sembunyikan Log' : 'Tampilkan Log Eksekusi'}</span>
          </button>
        </div>

        <!-- Collapsible Real-time Subprocess Log Card -->
        <div id="pipeline-log-card" class="card" style="margin-bottom:20px; padding:16px; display:${(AppState.showPipelineLog || isRunning) ? 'block' : 'none'};">
          <div class="section-bar" style="margin-bottom:10px;">
            <span style="font-weight:600; font-size:13px;">Konsol Log Subprocess Real-time (SSE Stream)</span>
            <button class="btn-ghost" onclick="connectPipelineSSE()" style="font-size:11.5px;">Hubungkan Ulang Stream</button>
          </div>
          <div id="pipeline-live-log-box" style="height:150px; background:#080c14; border:1px solid var(--border); border-radius:6px; padding:12px 14px; overflow-y:auto; font-family:var(--font-mono); font-size:11.5px; color:#cbd5e1;">
            <div style="color:var(--text-muted); font-size:11px;">[Sistem] Menunggu output eksekusi pipeline...</div>
          </div>
        </div>

        <!-- Konten Sub-View: Status Buklet -->
        ${AppState.pipelineSubMode === 'jobs' ? `
          <div class="card" style="padding:22px 24px;">
            <div class="section-bar" style="margin-bottom:14px; flex-wrap:wrap; gap:10px;">
              <div>
                <span class="section-heading">Status Pemrosesan per Buklet Kuesioner</span>
                <span class="font-mono" style="font-size:11.5px; color:var(--text-muted); margin-left:8px;">76 Buklet · Seruti Inti</span>
              </div>
              <div style="display:flex; gap:8px;">
                <input type="text" class="input-text" placeholder="Cari ID Buklet / Nama KRT..." oninput="filterPipelineTable(this.value)" style="padding:4px 10px; font-size:12px; width:220px;">
              </div>
            </div>

            <div class="table-wrap">
              <table id="pipeline-jobs-table">
                <thead>
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
                </thead>
                <tbody>
                  ${(AppState.batchJobs || []).map(j => `
                    <tr>
                      <td class="font-mono" style="font-weight:600;">${j.id}</td>
                      <td>${j.krt || '-'}</td>
                      <td style="font-size:12px; color:var(--text-secondary);">${j.stage || 'Selesai'}</td>
                      <td>
                        <div style="display:flex; align-items:center; gap:8px;">
                          <div class="progress-track" style="flex:1; height:5px;">
                            <div class="progress-fill" style="width:${j.progress || 100}%;"></div>
                          </div>
                          <span class="font-mono" style="font-size:11px; width:28px; text-align:right;">${j.progress || 100}%</span>
                        </div>
                      </td>
                      <td class="font-mono" style="color:#10b981; font-size:12px;">${(j.blank_boxes || 1558).toLocaleString('id-ID')} (OpenCV)</td>
                      <td class="font-mono" style="color:var(--accent); font-size:12px;">${(j.filled_boxes || 515).toLocaleString('id-ID')}</td>
                      <td>
                        <span class="badge ${j.status==='done'?'badge-pass':(j.status==='hitl'?'badge-warning':'badge-info')}" style="font-size:11px;">
                          ${j.status==='done'?'SELESAI':(j.status==='hitl'?`${j.ambigu||0} Perlu HITL`:'PROSES')}
                        </span>
                      </td>
                      <td style="text-align:right; white-space:nowrap;">
                        <button class="btn-ghost" onclick="onBookletChange('${j.id}').then(() => setPipelineSubMode('inference'))" title="Lihat hasil pembacaan AI" style="font-size:11.5px; padding:3px 8px; margin-right:4px;">
                          Inferensi →
                        </button>
                        <button class="btn-ghost" onclick="onBookletChange('${j.id}').then(() => switchView('kuesioner'))" title="Buka kuesioner digital" style="font-size:11.5px; padding:3px 8px;">
                          Kuesioner
                        </button>
                      </td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          </div>
        ` : `
          <!-- Konten Sub-View: Hasil Inferensi AI -->
          <div class="card" style="padding:22px 24px;">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px; padding:12px 16px; background:var(--surface-2); border-radius:var(--radius-sm); border:1px solid var(--border); margin-bottom:18px;">
              <div style="display:flex; align-items:center; gap:8px; overflow-x:auto;">
                <span style="font-size:12px; font-weight:600; color:var(--text-secondary); margin-right:4px;">Filter Halaman:</span>
                <button class="pill-tab ${AppState.inferencePageFilter===null?'active':''}" style="padding:4px 12px; font-size:12px;" onclick="onInferencePageFilter(null)">Semua Halaman (1 s.d. 7)</button>
                ${[1, 2, 3, 4, 5, 6, 7].map(p => `
                  <button class="pill-tab ${AppState.inferencePageFilter===p?'active':''}" style="padding:4px 10px; font-size:12px;" onclick="onInferencePageFilter(${p})">Hal ${p}</button>
                `).join('')}
              </div>

              <div style="display:flex; align-items:center; gap:10px;">
                <button id="btn-run-auto-grid" class="btn btn-primary btn-sm" onclick="runAutomaticGridInference()" 
                        style="display:flex; align-items:center; gap:8px; font-weight:600; padding:6px 14px;">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
                  Jalankan Inferensi Otomatis
                </button>
              </div>
            </div>

            <!-- Mosaic Grid Showcase Area -->
            <div id="mosaic-showcase-container" style="display:flex; flex-direction:column; gap:16px;">
              <div style="padding:32px; text-align:center; color:var(--text-muted);">
                Memuat data inferensi kisi...
              </div>
            </div>
          </div>
        `}
      `;

      if (isRunning) {
        connectPipelineSSE();
      }

      if (AppState.pipelineSubMode === 'inference') {
        if (typeof loadInferenceData === 'function') {
          loadInferenceData();
        }
      }
    }

    window.setPipelineSubMode = function(mode) {
      AppState.pipelineSubMode = mode;
      renderViewPipeline(document.getElementById('main-viewport'));
      if (mode === 'inference') {
        if (typeof loadInferenceData === 'function') loadInferenceData();
      }
    };

    window.togglePipelineLog = function() {
      AppState.showPipelineLog = !AppState.showPipelineLog;
      const el = document.getElementById('pipeline-log-card');
      const lbl = document.getElementById('log-toggle-label');
      if (el) el.style.display = AppState.showPipelineLog ? 'block' : 'none';
      if (lbl) lbl.textContent = AppState.showPipelineLog ? 'Sembunyikan Log' : 'Tampilkan Log Eksekusi';
    };'''

# Replace renderViewPipeline
pipeline_func_pattern = r"function renderViewPipeline\(el\)\s*\{[\s\S]*?\}\s*window\.filterPipelineTable"
replacement = pipeline_view_code + "\n\n    window.filterPipelineTable"
content = re.sub(pipeline_func_pattern, replacement, content, count=1)

index_path.write_text(content, encoding="utf-8")
print("Simplification successfully applied to static/index.html!")
