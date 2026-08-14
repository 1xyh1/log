@echo off
cd /d "%~dp0"

echo =========================================
echo Step 3 source mirror sync
echo Repo: 1xyh1/log
echo =========================================

python watch_step3_source_mirror.py

pause
