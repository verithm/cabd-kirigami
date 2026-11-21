# Runs every remaining state unattended, repeats what failed, and writes results\overnight_report.txt.
# Usage: run_overnight.bat [-Status]    (-Status only prints what is left)
# Closing the window is safe: Ansys stops with it and the next run continues where it stopped.
param([int]$WaitSec = 300, [int]$MaxStall = 3, [switch]$AllowBattery, [switch]$Status)
$ErrorActionPreference = "Stop"
$kit = $PSScriptRoot
$results = Join-Path $kit "results"
$report = Join-Path $results "overnight_report.txt"
$jobs = @(Import-Csv (Join-Path $kit "jobs.csv"))

function Read-Status($id, $s) {
    $f = Join-Path $results "$id\${s}_status.txt"
    if (-not (Test-Path $f)) { return "not_run" }
    $line = Get-Content $f | Where-Object { $_ -like "status=*" } | Select-Object -First 1
    if ($line) { return $line.Substring(7) }
    return "unreadable"
}

function All-States {
    foreach ($j in $jobs) {
        foreach ($s in ($j.states -split ';')) {
            $st = Read-Status $j.geom_id $s
            $curve = (Test-Path (Join-Path $results "$($j.geom_id)\${s}_curve.csv")) -or (Test-Path (Join-Path $results "$($j.geom_id)\${s}_autofailed_curve.csv"))
            $done = ($st -eq "ok" -or $st -eq "not_converged") -and $curve
            [pscustomobject]@{ id = $j.geom_id; state = $s; status = $st; done = $done }
        }
    }
}

function Write-Report($startedAt) {
    $all = @(All-States)
    $left = @($all | Where-Object { -not $_.done })
    $lines = @("Overnight run report", "started  : $startedAt", "finished : $(Get-Date -Format 'yyyy-MM-dd HH:mm')", "states   : $($all.Count)", "done     : $($all.Count - $left.Count)", "left     : $($left.Count)", "")
    foreach ($g in ($all | Group-Object status | Sort-Object Name)) { $lines += ("{0,-45} {1}" -f $g.Name, $g.Count) }
    if ($left.Count) {
        $lines += "", "NOT FINISHED:"
        foreach ($x in $left) { $lines += "$($x.id) $($x.state) $($x.status)" }
    }
    Set-Content -Path $report -Value $lines -Encoding ASCII
    $lines | ForEach-Object { Write-Host $_ }
    return $left.Count
}

if ($Status) { [void](Write-Report "(status only)"); exit 0 }

# Checks that can only be fixed by a person: everything must be right before leaving the PC alone.
$problems = @()
if (-not $env:AWP_ROOT261) { $problems += "Ansys 2026 R1 (AWP_ROOT261) not found." }
Add-Type -AssemblyName System.Windows.Forms
if (-not $AllowBattery -and [string][System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus -eq "Offline") { $problems += "The charger is not connected." }
$open = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(AnsysWBU|AnsysFWW?|RunWB2|ANSYS\d*)\.exe$' -and $_.CommandLine -notmatch '\s-b(\s|$)' })
if ($open.Count) { $problems += "Ansys is open: $(($open | ForEach-Object { $_.Name }) -join ', '). Close it." }
$other = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match '-File\s+"?[^"]*(collect|overnight)\.ps1' })
if ($other.Count) { $problems += "Another collection window is running (PID $($other[0].ProcessId)). Close it." }
$freeGB = [math]::Round((Get-PSDrive ($kit.Substring(0, 1))).Free / 1GB, 1)
if ($freeGB -lt 15) { $problems += "Only $freeGB GB free on the kit drive; 15 GB or more is needed." }
if ($problems.Count) {
    Write-Host "NOT STARTED. Fix these and run again:" -ForegroundColor Red
    $problems | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}

$startedAt = Get-Date -Format "yyyy-MM-dd HH:mm"
$stall = 0
$round = 0
while ($true) {
    $before = @(All-States | Where-Object { -not $_.done }).Count
    if ($before -eq 0) { break }
    $round++
    Write-Host ""
    Write-Host "===== round $round | $before states left | $(Get-Date -Format 'MM-dd HH:mm') =====" -ForegroundColor Cyan
    # Child errors must not stop this loop: the state counts below decide what happens next.
    $ErrorActionPreference = "Continue"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $kit "collect.ps1") -RetryFailed -NoZip
    $ErrorActionPreference = "Stop"
    $after = @(All-States | Where-Object { -not $_.done }).Count
    if ($after -eq 0) { break }
    if ($after -lt $before) { $stall = 0 } else { $stall++ }
    if ($stall -ge $MaxStall) {
        Write-Host "No progress in $MaxStall rounds in a row. Stopping." -ForegroundColor Red
        break
    }
    Write-Host "$after states left. Waiting $WaitSec s for licenses and processes to clear."
    Start-Sleep -Seconds $WaitSec
}

Write-Host ""
$left = Write-Report $startedAt
& (Join-Path $kit "make_result_zip.ps1")
if ($left -eq 0) { Write-Host "ALL DONE." -ForegroundColor Green; exit 0 }
Write-Host "$left states are not finished. See $report" -ForegroundColor Yellow
exit 1
