@echo off
rem Double-click to start or continue. Closing this window stops Ansys safely.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0overnight.ps1" %*
pause
