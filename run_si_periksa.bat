@echo off
title Si-Periksa Seruti Scanner v2.0 — BPS Penajam Paser Utara
echo ========================================================
echo   Si-Periksa v2.0: Sistem Pembacaan Dokumen Survei
echo   Badan Pusat Statistik Kabupaten Penajam Paser Utara
echo ========================================================
echo.
echo Mode: Lokal (127.0.0.1:8000)
echo Membuka server...
cd /d "%~dp0"
start http://127.0.0.1:8000
python app.py
pause
