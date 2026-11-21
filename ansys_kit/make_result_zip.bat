@echo off
rem Zip the results collected so far (can be run while run_collect.bat is running).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_result_zip.ps1"
pause
