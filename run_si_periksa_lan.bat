@echo off
title Si-Periksa Seruti Scanner v2.0 [LAN MODE] — BPS Penajam Paser Utara
echo ========================================================
echo   Si-Periksa v2.0: Sistem Pembacaan Dokumen Survei
echo   MODE LAN / JARINGAN LOKAL (0.0.0.0:8000)
echo   Badan Pusat Statistik Kabupaten Penajam Paser Utara
echo ========================================================
echo.
echo Mode: LAN (Dapat diakses oleh PC pengawas / petugas lain)
cd /d "%~dp0"
start http://127.0.0.1:8000
python app.py --lan
pause
