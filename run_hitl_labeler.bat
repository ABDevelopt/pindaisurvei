@echo off
title BPS Seruti Inti - HITL Labeling Verifier
echo ======================================================================
echo   BPS SERUTI INTI - HUMAN-IN-THE-LOOP (HITL) LABELING VERIFIER
echo ======================================================================
echo Memulai server verifikasi manual pada http://localhost:8585 ...
python seruti_pipeline\hitl_labeler_server.py --port 8585
pause
