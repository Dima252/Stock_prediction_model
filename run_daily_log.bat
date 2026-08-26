@echo off
REM Daily forward-test logger + site rebuild for market_drop.
REM Paper only - places no orders. Schedule ~30 min after the US close.
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\13_forward_log.py"  >> "data\live\run.log" 2>&1
".venv\Scripts\python.exe" "scripts\14_build_site.py"   >> "data\live\run.log" 2>&1
