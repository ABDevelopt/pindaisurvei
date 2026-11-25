@echo off
title BPS Seruti - HITL Annotation Verifier (Opsi A)
echo ======================================================================
echo   BPS SERUTI - HUMAN-IN-THE-LOOP (HITL) ANNOTATION VERIFIER
echo   Target: 660 Citra Prioritas Tinggi (Tier 1 & Tier 2)
echo ======================================================================
echo Memulai server verifikasi manual pada:
echo   - Laptop : http://localhost:8088
echo   - HP     : http://10.108.181.235:8088
echo ======================================================================
python seruti_hitl_server.py 8088
pause
