@echo off
REM Daily forward-test logger, site rebuild, and publish. Paper only - no orders.
REM Scheduled Tue-Sat 00:30 local (after the US close under the 21:00 UTC guard).
cd /d "%~dp0"
set LOG=data\live\run.log

".venv\Scripts\python.exe" "scripts\13_forward_log.py"  >> "%LOG%" 2>&1
".venv\Scripts\python.exe" "scripts\14_build_site.py"   >> "%LOG%" 2>&1

REM Publish the rebuilt page. Only commits when docs/ actually changed, so a
REM day with no new data leaves no empty commit.
git diff --quiet -- docs || (
  git add docs/index.html docs/.nojekyll
  git commit -q -m "site: snapshot %DATE%" >> "%LOG%" 2>&1
  git push -q origin main >> "%LOG%" 2>&1
)
