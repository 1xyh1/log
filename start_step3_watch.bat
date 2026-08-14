@echo off
cd /d "%~dp0"

echo ==========================================
echo Step 3 code/log Git watcher
echo ==========================================

".venv\Scripts\python.exe" "tools\auto_git_watch.py"

pause
