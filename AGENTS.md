# Aturan Mutlak Desain Antarmuka: Kesederhanaan Radikal & Ruang Kosong (Clean & Breathing UI)

Setiap pembuatan atau modifikasi antarmuka visual (wireframe, HTML, CSS, React, dashboard, atau prototipe) pada repositori ini **WAJIB MEMATUHI** prinsip berikut:

## 1. Larangan Tampilan Ramai (Anti-Clutter Invariant)
- **Dilarang Menumpuk Elemen**: Jangan menjejalkan terlalu banyak kartu statistik, badge warna-warni, border tebal, atau panel dalam satu layar.
- **Maksimal 3-4 Metrik Utama**: Pada dashboard atau ringkasan, tampilkan hanya metrik yang paling krusial. Metrik sekunder disembunyikan dalam drill-down atau tab terpisah.
- **Batasi Badge & Tag**: Jangan memberi badge status pada setiap baris teks. Gunakan badge secara selektif hanya untuk anomali/perhatian khusus.

## 2. Rangkul Ruang Kosong (Embrace Generous Negative Space)
- **Jangan Takut Ruang Kosong**: Ruang kosong (white/dark space) adalah ruang bernapas visual (*visual breathing room*), bukan ruang terbuang yang harus diisi.
- **Margin & Padding yang Lega**: Gunakan padding dan gap yang leluasa (minimal 24px - 32px antar-seksi besar, 16px antar-elemen).
- **Fokus Tunggal per Layar**: Setiap layar atau panel harus memiliki satu hierarki visual utama yang langsung tampak jelas dalam 3 detik pertama.

## 3. Selalu Lakukan Penyederhanaan (Radical Simplification)
- Jika suatu kontrol, tombol, atau teks penjelasan tidak esensial untuk menyelesaikan alur kerja saat itu, **HAPUS** atau pindahkan ke menu sekunder/modal bantuan.
- Kurangi garis batas (*borders*) yang berlebihan; gunakan kontras warna latar belakang lembut untuk memisahkan area, bukan garis kotak bertumpuk-tumpuk.

## 4. Larangan Mutlak Penggunaan Emoji / Emoticon sebagai Ikon (Wajib Iconpack / SVG Asli)
- **Dilarang Keras Menggunakan Karakter Emoji Sebagai Ikon**: Dilarang menggunakan simbol emoji Unicode (seperti 📖, 📄, ✂️, 📕, 🔍, ⚠️, ✅, ⚙️, dsb.) atau emoticon teks sebagai ikon tombol, tab navigasi, label status, atau placeholder visual.
- **Wajib Menggunakan Iconpack Vektor Resmi**: Gunakan format SVG vektor terstandar (misalnya Lucide Icons, Heroicons, atau Phosphor Icons) dengan ukuran proporsional (14px–18px untuk tombol/tab, 24px–32px untuk ilustrasi kosong), `stroke-width="1.75"` atau `2`, `stroke="currentColor"`, dan `fill="none"`.
- **Konsistensi Gaya Visual**: Seluruh ikon harus memiliki ketebalan goresan (*stroke weight*), kelengkungan sudut (*corner radius*), dan warna yang seragam mengikuti token desain antarmuka (`currentColor` / CSS variable).
