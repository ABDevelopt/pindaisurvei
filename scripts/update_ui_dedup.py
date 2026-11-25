with open("static/index.html", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update dropdown
text = text.replace(
    '<option value="seruti_inti">VSERUTI26.INTI (76)</option>',
    '<option value="seruti_inti">VSERUTI26.INTI · Seruti Inti (51)</option>'
)
text = text.replace(
    '<option value="seruti_inti">VSERUTI26.INTI · Seruti Inti (76)</option>',
    '<option value="seruti_inti">VSERUTI26.INTI · Seruti Inti (51)</option>'
)

# 2. Update View 1 (Input)
text = text.replace(
    "showToast('Batch 76 buklet dikirim ke Pipeline Produksi.', 'info');",
    "showToast('Batch 51 buklet unik dikirim ke Pipeline Produksi.', 'info');"
)
text = text.replace(
    '<div class="metric-value">76</div>\n            <div class="metric-label"><span class="status-dot dot-green"></span>Buklet Lengkap</div>',
    '<div class="metric-value">51</div>\n            <div class="metric-label"><span class="status-dot dot-green"></span>Buklet Unik Lengkap</div>'
)

# 3. Update Pipeline View
text = text.replace("const totalBoxes = 95358;", "const totalBoxes = 63954;")
text = text.replace("const blankBoxes = 74699;", "const blankBoxes = 50076;")
text = text.replace("const filledBoxes = 20659;", "const filledBoxes = 13878;")
text = text.replace("'Selesai (76 Buklet)'", "'Selesai (51 Buklet Unik)'")
text = text.replace("Pipeline Siap · 76 Buklet Terakreditasi BPS", "Pipeline Siap · 51 Buklet Unik Terakreditasi BPS")
text = text.replace("Jalankan Seluruh 76 Buklet", "Jalankan Seluruh 51 Buklet Unik")
text = text.replace("76 Buklet · Hal 1 s.d. Hal 7", "51 Buklet Unik · Hal 1 s.d. Hal 7")
text = text.replace("Status Pemrosesan Buklet (76)", "Status Pemrosesan Buklet (51)")
text = text.replace("76 Buklet · Seruti Inti", "51 Buklet Unik · Seruti Inti")

with open("static/index.html", "w", encoding="utf-8") as f:
    f.write(text)

print("static/index.html updated for 51 unique booklets!")
