# Zip everything in results\ into one file to send to the team (safe to run while collect is running).
$kit = $PSScriptRoot
$results = Join-Path $kit "results"
$cfg = Get-Content (Join-Path $kit "kit_config.json") -Raw | ConvertFrom-Json
$done = @(Get-ChildItem $results -Recurse -Filter "*_status.txt" -ErrorAction SilentlyContinue).Count
$zip = Join-Path $kit ("collect_result_PC{0}_{1}_{2}.zip" -f $cfg.pc, $env:COMPUTERNAME, (Get-Date -Format "yyyyMMdd_HHmm"))
Compress-Archive -Path "$results\*" -DestinationPath $zip -Force
Write-Host ""
Write-Host "finished states in results: $done"
Write-Host "Send this file to the team: $zip" -ForegroundColor Green
