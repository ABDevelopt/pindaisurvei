with open('static/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''              <div style="display:flex; align-items:center; gap:10px;">
                <button id="btn-run-auto-grid" class="btn btn-primary btn-sm" onclick="runAutomaticGridInference()"'''

replacement = '''              <div style="display:flex; align-items:center; gap:10px;">
                <button id="btn-gemini-config" class="pill-tab" onclick="openGeminiConfigModal()" title="Konfigurasi Google Gemini API Key"
                        style="display:flex; align-items:center; gap:6px; padding:6px 12px; font-size:12px; border:1px solid var(--border); background:var(--surface-1); cursor:pointer;">
                  <span id="gemini-status-dot" class="status-dot dot-gray"></span>
                  <span id="gemini-status-label" class="font-mono">Gemini API</span>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
                </button>
                <button id="btn-run-auto-grid" class="btn btn-primary btn-sm" onclick="runAutomaticGridInference()"'''

if target in text:
    text = text.replace(target, replacement, 1)

target2 = '''      if (AppState.pipelineSubMode === 'inference') {
        if (typeof loadInferenceData === 'function') {
          loadInferenceData();
        }
      }'''

replacement2 = '''      if (AppState.pipelineSubMode === 'inference') {
        if (typeof loadGeminiStatus === 'function') {
          loadGeminiStatus();
        }
        if (typeof loadInferenceData === 'function') {
          loadInferenceData();
        }
      }'''

if target2 in text:
    text = text.replace(target2, replacement2, 1)

with open('static/index.html', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated inference subview with Gemini config button & loader')
