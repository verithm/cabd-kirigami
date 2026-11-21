@echo off
rem Double-click to start (or continue) the data collection. Optional args: -MeshOnly, -Limit N, -RetryFailed
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0collect.ps1" %*
pause
