# Data-collection driver: runs every job in jobs.csv (one geometry = slightly wet + fully wet),
# extracts the raw curves with MAPDL, keeps only the small result files, and can be stopped/restarted
# at any time (finished states are skipped).
# Usage: run_collect.bat [-MeshOnly] [-Limit N] [-RetryFailed]
param([switch]$MeshOnly, [int]$Limit = 0, [switch]$RetryFailed, [string]$PostDir = "", [string]$PostId = "", [string]$PostState = "")
$ErrorActionPreference = "Stop"
$kit = $PSScriptRoot

if ($kit -match '[^\x21-\x7E\\:]') {
    Write-Host "ERROR: the kit folder path must contain only English letters/digits and no spaces." -ForegroundColor Red
    Write-Host "       Current path: $kit"
    Write-Host "       Move the folder to e.g. C:\CABD_collect and run again."
    exit 1
}
$root = $env:AWP_ROOT261
if (-not $root) { Write-Host "ERROR: Ansys 2026 R1 (AWP_ROOT261) not found." -ForegroundColor Red; exit 1 }
$runwb = Join-Path $root "Framework\bin\Win64\RunWB2.exe"
$mapdl = Join-Path $root "ansys\bin\winx64\ANSYS261.exe"

$cfg = Get-Content (Join-Path $kit "kit_config.json") -Raw | ConvertFrom-Json
$pc = $cfg.pc
$results = Join-Path $kit "results"
New-Item -ItemType Directory -Force $results | Out-Null
$progress = Join-Path $results "progress.csv"
if (-not (Test-Path $progress)) {
    Set-Content -Path $progress -Value "finished,pc,geom_id,state,status,mode,mesh_rule,nodes,n_sets,last_time,job_min" -Encoding ASCII
}

# Keep the PC awake while this window runs (released automatically when it closes).
Add-Type -Namespace Cabd -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Cabd.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

# Personal: put this window and everything it starts in a Windows job. Windows kills them all when this window ends for any reason.
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
namespace Cabd {
    public static class KillOnExit {
        [StructLayout(LayoutKind.Sequential)] struct Basic { public long a, b; public uint LimitFlags; public UIntPtr MinWs, MaxWs; public uint ProcLimit; public UIntPtr Affinity; public uint Priority, Sched; }
        [StructLayout(LayoutKind.Sequential)] struct Io { public ulong a, b, c, d, e, f; }
        [StructLayout(LayoutKind.Sequential)] struct Ext { public Basic B; public Io I; public UIntPtr ProcMem, JobMem, PeakProc, PeakJob; }
        [DllImport("kernel32.dll")] static extern IntPtr CreateJobObject(IntPtr attr, string name);
        [DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr job, int cls, ref Ext info, uint len);
        [DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr proc);
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
        static IntPtr job;
        public static bool Enable() {
            job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) return false;
            Ext e = new Ext();
            e.B.LimitFlags = 0x2000;  // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            return SetInformationJobObject(job, 9, ref e, (uint)Marshal.SizeOf(typeof(Ext))) && AssignProcessToJobObject(job, GetCurrentProcess());
        }
    }
}
"@
if (-not [Cabd.KillOnExit]::Enable()) { Write-Host "WARNING: could not enable automatic Ansys shutdown for this window." -ForegroundColor Yellow }

Add-Type -AssemblyName System.Windows.Forms
function PowerState { [string][System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus }

$machine = Join-Path $results "machine_$($env:COMPUTERNAME).txt"
if (-not (Test-Path $machine)) {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
    @("kit_pc      : $pc", "computer    : $($env:COMPUTERNAME)", "cpu         : $($cpu.Name.Trim()) | cores $($cpu.NumberOfCores)",
      "ram_gb      : $ramGB", "ansys_root  : $root", "kit_version : $($cfg.kit_version)") | Set-Content -Path $machine -Encoding UTF8
}

$jobs = @(Import-Csv (Join-Path $kit "jobs.csv"))
$landmarks = @{}
foreach ($row in Import-Csv (Join-Path $kit "landmarks.csv")) {
    if (-not $landmarks.ContainsKey($row.geom_id)) { $landmarks[$row.geom_id] = New-Object System.Collections.ArrayList }
    [void]$landmarks[$row.geom_id].Add($row)
}
$BODY_NAMES = @("frame", "arm_upper", "arm_lower", "hinge_tip_upper", "hinge_tip_lower", "hinge_apex")
# PowerShell names are case-insensitive: this must not share a name with $states of each job below
$ALL_STATES = @("partial", "full")

function Kill-Tree($proc) {
    if ($proc -and -not $proc.HasExited) { & taskkill.exe /T /F /PID $proc.Id | Out-Null }
}

function Run-Process($exe, $argList, $dir, $timeoutMin) {
    $p = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $dir -PassThru -NoNewWindow
    if (-not $p.WaitForExit([int]($timeoutMin * 60000))) { Kill-Tree $p; return $false }
    return $true
}

function Ansys-Processes {
    @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(AnsysWBU|AnsysFWW?|RunWB2|ANSYS\d*)\.exe$' })
}

function Stop-LeftoverAnsys {
    # A closed window can leave a batch Mechanical behind. It keeps the only solver license, so every later solve fails to start.
    $left = @(Ansys-Processes | Where-Object { $_.CommandLine -match '\s-b(\s|$)' })
    foreach ($p in $left) {
        Write-Host ("Stopping leftover {0} (PID {1}, started {2})" -f $p.Name, $p.ProcessId, $p.CreationDate) -ForegroundColor Yellow
        & taskkill.exe /T /F /PID $p.ProcessId 2>$null | Out-Null
    }
    if ($left.Count) { Start-Sleep -Seconds 30 }  # let the license server take the license back
}

function Body-Materials($dsdat) {
    # Map body name -> MAT id from the "Elements for Body" blocks Mechanical writes into ds.dat.
    $map = @{}
    $lines = [System.IO.File]::ReadAllLines($dsdat)
    for ($i = 0; $i -lt $lines.Length; $i++) {
        if ($lines[$i] -match "Elements for Body\s+\d+\s+['`"]([^'`"|]+)") {
            $name = $Matches[1].Trim()
            for ($j = $i + 1; $j -lt [math]::Min($i + 10, $lines.Length); $j++) {
                # Mechanical writes "MAT,4 $ TYPE,4 $ ..." right after the body header (eblock is COMPACT: no MAT field)
                if ($lines[$j] -match '^\s*MAT,(\d+)') {
                    if ($BODY_NAMES -contains $name) { $map[$name] = [int]$Matches[1] }
                    break
                }
            }
        }
    }
    return $map
}

function Completed-Time($dir) {
    # time of the last converged substep = column 7 of file.mntr (same rule as mech_job.py); no file = no filtering
    $mntr = Join-Path $dir "file.mntr"
    if (-not (Test-Path $mntr)) { return 1e9 }
    $done = 0.0
    foreach ($line in Get-Content $mntr) {
        $f = $line.Trim() -split '\s+'
        if ($f.Count -ge 7 -and $f[0] -match '^\d+$' -and $f[1] -match '^\d+$') { $done = [double]$f[6] }
    }
    return $done
}

function Try-Extract($id, $state, $dir, $out) {
    # a post-processing failure is recorded for this state; it must never stop the whole collection
    try { return (Extract-State $id $state $dir $out) } catch { return "post_error: $($_.Exception.Message)" }
}

function Extract-State($id, $state, $dir, $out) {
    $mats = Body-Materials (Join-Path $dir "ds.dat")
    if ($mats.Count -ne 6) { return "post_error: body materials not found ($($mats.Count))" }
    $lm = $landmarks[$id]
    $p = @("NBOD=6", "*DIM,BMAT,ARRAY,6")
    for ($b = 0; $b -lt 6; $b++) { $p += "BMAT($($b + 1))=$($mats[$BODY_NAMES[$b]])" }
    $p += "TMAX=$((Completed-Time $dir) + 1e-4)", "NLM=$($lm.Count)", "LMY=6", "*DIM,LMX,ARRAY,NLM", "*DIM,LMZ,ARRAY,NLM"
    for ($i = 0; $i -lt $lm.Count; $i++) { $p += "LMX($($i + 1))=$($lm[$i].x)"; $p += "LMZ($($i + 1))=$($lm[$i].z)" }
    Set-Content -Path (Join-Path $dir "post_params.inp") -Value $p -Encoding ASCII
    foreach ($f in "sets.txt", "bodies.txt", "lm.txt", "lm_nodes.txt") { Remove-Item (Join-Path $dir $f) -ErrorAction SilentlyContinue }
    Copy-Item (Join-Path $kit "post_job.inp") $dir -Force
    $okPost = Run-Process $mapdl @("-b", "-i", "post_job.inp", "-o", "post.out", "-j", "postjob") $dir $cfg.post_timeout_min
    Copy-Item (Join-Path $dir "post.out") (Join-Path $out "${state}_post.out") -ErrorAction SilentlyContinue
    if (-not $okPost -or -not (Test-Path (Join-Path $dir "sets.txt"))) { return "post_error: MAPDL post failed" }

    $time = @{}
    $curve = @{}
    foreach ($r in Get-Content (Join-Path $dir "sets.txt")) {
        $v = $r -split ',' | ForEach-Object { $_.Trim() }
        $k = [int][double]$v[0]
        $time[$k] = $v[1]
        $curve[$k] = New-Object System.Collections.ArrayList
        [void]$curve[$k].AddRange(@($v[1], $v[2], $v[3], $v[4], $v[5]))
    }
    $perBody = @{}
    foreach ($r in Get-Content (Join-Path $dir "bodies.txt")) {
        $v = $r -split ',' | ForEach-Object { $_.Trim() }
        $perBody["$([int][double]$v[0])_$([int][double]$v[1])"] = $v
    }
    $hdr = "set,time,u_applied_mm,fx_disp_N,fx_fix_N,U_total_mJ,seqv_max_MPa,seqv_max_body,seqv_max_x_mm,seqv_max_z_mm"
    foreach ($b in $BODY_NAMES) { $hdr += ",U_${b}_mJ" }
    foreach ($b in $BODY_NAMES) { $hdr += ",seqv_max_${b}_MPa" }
    $rows = @($hdr)
    foreach ($k in ($curve.Keys | Sort-Object)) {
        if (@(1..6 | Where-Object { -not $perBody.ContainsKey("${k}_$_") }).Count) { continue }  # MAPDL stopped inside this set
        $best = -1.0; $bestB = ""; $bx = ""; $bz = ""; $u = @(); $s = @()
        for ($b = 1; $b -le 6; $b++) {
            $v = $perBody["${k}_$b"]
            $u += $v[2]; $s += $v[3]
            if ([double]$v[3] -gt $best) { $best = [double]$v[3]; $bestB = $BODY_NAMES[$b - 1]; $bx = $v[5]; $bz = $v[6] }
        }
        $rows += (@($k) + $curve[$k] + @($best.ToString("E8"), $bestB, $bx, $bz) + $u + $s) -join ","
    }
    Set-Content -Path (Join-Path $out "${state}_curve.csv") -Value $rows -Encoding ASCII

    $nodeOf = @{}
    $lmRows = @("landmark,name,x0_mm,z0_mm,node,node_x_mm,node_y_mm,node_z_mm")
    foreach ($r in Get-Content (Join-Path $dir "lm_nodes.txt")) {
        $v = $r -split ',' | ForEach-Object { $_.Trim() }
        $i = [int][double]$v[0]
        $lmRows += "$i,$($lm[$i - 1].name),$($lm[$i - 1].x),$($lm[$i - 1].z),$([int][double]$v[1]),$($v[2]),$($v[3]),$($v[4])"
    }
    Set-Content -Path (Join-Path $out "${state}_landmark_nodes.csv") -Value $lmRows -Encoding ASCII
    $dRows = @("set,time,landmark,ux_mm,uy_mm,uz_mm")
    foreach ($r in Get-Content (Join-Path $dir "lm.txt")) {
        $v = $r -split ',' | ForEach-Object { $_.Trim() }
        $k = [int][double]$v[0]
        $dRows += "$k,$($time[$k]),$([int][double]$v[1]),$($v[2]),$($v[3]),$($v[4])"
    }
    Set-Content -Path (Join-Path $out "${state}_landmark_disp.csv") -Value $dRows -Encoding ASCII
    return "ok"
}

if ($PostDir) {
    # re-extract one solved state from an existing solver directory (debugging)
    $out = Join-Path $results $PostId
    New-Item -ItemType Directory -Force $out | Out-Null
    Write-Host (Try-Extract $PostId $PostState $PostDir $out)
    exit 0
}

function Pending-States($id) {
    $out = Join-Path $results $id
    $todo = @()
    foreach ($s in $ALL_STATES) {
        $st = Join-Path $out "${s}_status.txt"
        if (-not (Test-Path $st)) { $todo += $s; continue }
        # the solver never started: not a property of the geometry, so it always runs again
        if (Select-String -Path $st -Pattern "^status=(start_error|license_error)$" -Quiet) { $todo += $s; continue }
        # not_converged is a real result: the same mesh and settings would fail the same way
        if ($RetryFailed -and -not (Select-String -Path $st -Pattern "^status=(ok|not_converged)$" -Quiet)) { $todo += $s }
    }
    return $todo
}

$other = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match '-File\s+"?[^"]*collect\.ps1' })
if ($other.Count) {
    Write-Host "ERROR: another collection window is already running on this PC (PID $($other[0].ProcessId)). Close it first." -ForegroundColor Red
    exit 1
}
Stop-LeftoverAnsys
$open = @(Ansys-Processes)
if ($open.Count) {
    Write-Host "ERROR: Ansys is open: $(($open | ForEach-Object { $_.Name }) -join ', '). Save your work, close it, and run again (or reboot)." -ForegroundColor Red
    exit 1
}

$todoJobs = @($jobs | Where-Object { (Pending-States $_.geom_id).Count -gt 0 })
if ($Limit -gt 0) { $todoJobs = @($todoJobs | Select-Object -First $Limit) }
$doneBefore = $jobs.Count - @($jobs | Where-Object { (Pending-States $_.geom_id).Count -gt 0 }).Count
Write-Host "== CABD data collection | kit PC$pc | $($env:COMPUTERNAME) =="
Write-Host "jobs in kit: $($jobs.Count) | already finished: $doneBefore | to run now: $($todoJobs.Count)"
if ((PowerState) -eq "Offline") { Write-Host "WARNING: running on battery. Plug in the charger (about 1.7x slower on battery)." -ForegroundColor Yellow }
Write-Host "Keep this window open. You can stop with Ctrl+C or by closing it; the next run cleans up and continues where it stopped."
Write-Host ""

$MAX_START_FAILS = 3
$startFails = 0
$n = 0
$times = New-Object System.Collections.ArrayList
foreach ($job in $todoJobs) {
    $n++
    $id = $job.geom_id
    Stop-LeftoverAnsys
    $states = @(Pending-States $id)
    $out = Join-Path $results $id
    New-Item -ItemType Directory -Force $out | Out-Null
    $work = Join-Path $kit "work\$id"
    if (Test-Path $work) { Remove-Item -Recurse -Force $work }
    New-Item -ItemType Directory -Force $work | Out-Null
    $meshOnlyPy = if ($MeshOnly) { "True" } else { "False" }
    @(
        "GEOM_ID = `"$id`"", "X_W = $($job.x_fixed_face)", "X_D = $($job.x_displaced_face)",
        "T_H = $($job.t_h)", "T_A = $($job.t_a)",
        "GLOBAL_SIZE = $($cfg.global_size_mm)", "HINGE_MESH_ATTEMPTS = $($cfg.hinge_mesh_attempts)", "RETRY_MESH = $($cfg.retry_mesh)",
        "NODE_LIMIT = $($cfg.node_limit)", "MESH_ONLY = $meshOnlyPy"
    ) | Set-Content -Path (Join-Path $work "job_header.py") -Encoding ASCII

    $stamp = Get-Date -Format "MM-dd HH:mm"
    Write-Host ("[{0}] job {1}/{2}  {3}  states: {4}" -f $stamp, $n, $todoJobs.Count, $id, ($states -join ","))
    $env:JOB_KIT = $kit
    $env:JOB_WORK = $work
    $env:JOB_STEP = Join-Path $kit "step\$id.step"
    $env:JOB_STATES = $states -join ","
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $okWb = Run-Process $runwb @("-B", "-R", "`"$kit\job.wbjn`"") $work ($cfg.job_timeout_min_per_state * $states.Count)
    $wbLog = Join-Path $work "wb_log.txt"
    if (Test-Path $wbLog) { Add-Content -Path (Join-Path $out "wb_log.txt") -Value (Get-Content $wbLog) -Encoding UTF8 }
    $log = if (Test-Path $wbLog) { Get-Content $wbLog } else { @() }
    $noStart = $true

    foreach ($state in $states) {
        $g = { param($pat) ($log | Select-String -Pattern "^$state $pat" | Select-Object -Last 1) }
        $info = [ordered]@{ geom_id = $id; state = $state; pc = $pc; computer = $env:COMPUTERNAME; status = ""; mode = ""; mesh_rule = ""; nodes = ""; n_sets = ""; last_time = "" }
        $m = & $g "mesh_rule=(\S+) nodes=(\d+) elements=(\d+)"
        if ($m) { $info.mesh_rule = $m.Matches[0].Groups[1].Value; $info.nodes = $m.Matches[0].Groups[2].Value; $info.elements = $m.Matches[0].Groups[3].Value }
        $sm = @($log | Select-String -Pattern "^$state solve mode=(\S+) .*completed_time=([\d.]+)")
        if ($sm.Count) {
            $info.completed_time = $sm[-1].Matches[0].Groups[2].Value
            if ($sm[0].Matches[0].Groups[1].Value -eq "auto") { $info.auto_completed_time = $sm[0].Matches[0].Groups[2].Value }
        }
        $fm = & $g "final_mode=(\S+) ok=(\S+)"
        if ($fm) { $info.mode = $fm.Matches[0].Groups[1].Value }
        $err = & $g "ERROR (.+)"
        $dirLine = & $g "solver_dir=(.+)$"
        if ($MeshOnly) {
            $info.status = if ($info.nodes) { "mesh_only" } else { "mesh_error" }
        } elseif (-not $okWb) {
            $info.status = "timeout"
        } elseif ($err) {
            $info.status = "error: $($err.Matches[0].Groups[1].Value)"
        } elseif (-not $dirLine) {
            $info.status = "error: no solver directory (see wb_log.txt)"
        } else {
            $dir = $dirLine.Matches[0].Groups[1].Value.Trim()
            if (-not (Test-Path (Join-Path $dir "file.rst"))) {
                $info.status = "error: no result file"
            } else {
                $info.status = Try-Extract $id $state $dir $out
                $converged = ($fm -and $fm.Matches[0].Groups[2].Value -eq "True")
                if ($info.status -eq "ok" -and -not $converged) { $info.status = "not_converged" }
                Copy-Item (Join-Path $dir "solve.out") (Join-Path $out "${state}_solve.out") -ErrorAction SilentlyContinue
                $curveFile = Join-Path $out "${state}_curve.csv"
                if (Test-Path $curveFile) {
                    $c = @(Import-Csv $curveFile)
                    $info.n_sets = $c.Count
                    if ($c.Count) { $info.last_time = $c[-1].time }
                }
            }
            # a lost license is not a property of the geometry: flagged so -RetryFailed can run it again
            $fmOk = ($fm -and $fm.Matches[0].Groups[2].Value -eq "True")
            if (-not $fmOk -and ($log | Select-String -Pattern "^$state message Error: .*license" -Quiet)) { $info.status = "license_error" }
            elseif (-not $fmOk -and ($log | Select-String -Pattern "^$state solver_start_failed" -Quiet)) { $info.status = "start_error" }
            # the failed auto-time-stepping attempt is kept too (curve up to the failure)
            $af = & $g "auto_failed_dir=(.+)$"
            if ($af) {
                $adir = $af.Matches[0].Groups[1].Value.Trim()
                if (Test-Path (Join-Path $adir "file.rst")) {
                    $info.auto_failed_extract = Try-Extract $id "${state}_autofailed" $adir $out
                    Copy-Item (Join-Path $adir "solve.out") (Join-Path $out "${state}_autofailed_solve.out") -ErrorAction SilentlyContinue
                    $ac = Join-Path $out "${state}_autofailed_curve.csv"
                    if (Test-Path $ac) { $c = @(Import-Csv $ac); if ($c.Count) { $info.auto_failed_last_time = $c[-1].time } }
                }
            }
        }
        $info.job_min = [math]::Round($sw.Elapsed.TotalMinutes, 1)
        $info.finished = Get-Date -Format "yyyy-MM-dd HH:mm"
        ($info.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) | Set-Content -Path (Join-Path $out "${state}_status.txt") -Encoding ASCII
        Add-Content -Path $progress -Value ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}" -f $info.finished, $pc, $id, $state, ($info.status -replace ',', ';'), $info.mode, $info.mesh_rule, $info.nodes, $info.n_sets, $info.last_time, $info.job_min) -Encoding ASCII
        if ($info.status -ne "start_error" -and $info.status -ne "license_error") { $noStart = $false }
        $color = if ($info.status -eq "ok" -or $info.status -eq "mesh_only") { "Gray" } else { "Yellow" }
        Write-Host ("          {0,-8} {1} | mode {2} | nodes {3} | substeps {4} | last time {5}" -f $state, $info.status, $info.mode, $info.nodes, $info.n_sets, $info.last_time) -ForegroundColor $color
    }
    $sw.Stop()
    if (-not $MeshOnly) { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
    [void]$times.Add($sw.Elapsed.TotalMinutes)
    $avg = ($times | Measure-Object -Average).Average
    $left = $todoJobs.Count - $n
    Write-Host ("          {0:F1} min | average {1:F1} min/job | {2} jobs left, about {3:F1} h" -f $sw.Elapsed.TotalMinutes, $avg, $left, ($left * $avg / 60))
    if ((PowerState) -eq "Offline") { Write-Host "          WARNING: on battery" -ForegroundColor Yellow }
    $startFails = if ($noStart) { $startFails + 1 } else { 0 }
    if ($startFails -ge $MAX_START_FAILS) {
        Write-Host ""
        Write-Host "STOPPED: the solver could not start for $MAX_START_FAILS geometries in a row. This is not a geometry problem." -ForegroundColor Red
        Write-Host "Reboot the PC, close other programs, and run run_collect.bat again. These geometries run again automatically." -ForegroundColor Red
        exit 1
    }
}

& (Join-Path $kit "make_result_zip.ps1")
