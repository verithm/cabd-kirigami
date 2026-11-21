@echo off
rem Starts the CABD demo app at http://localhost:8765 and opens the browser. Close this window to stop it.
rem Uses the python on PATH (or %CABD_PYTHON% if set) with numpy, pandas, scipy, scikit-learn, joblib and torch.
cd /d "%~dp0.."
set PY=python
if defined CABD_PYTHON set PY=%CABD_PYTHON%
set PYTHONPATH=%CD%\src
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8765"
"%PY%" -X utf8 -m cabd.app.server --port 8765
pause
