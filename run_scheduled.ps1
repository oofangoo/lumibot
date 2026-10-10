# Usage: .\run_scheduled.ps1 stocks|btc
# Scheduler entry point: runs one LumiBot live-paper iteration and appends everything it prints to logs\lumi_<group>.log,
# plus a one-line status record in logs\run_status.csv (time, group, exit code, seconds, code version) so "ran, no trade" and
# "did not run" differ, and every run can be traced to the exact commit that produced it.
param([Parameter(Mandatory = $true)][ValidateSet('stocks', 'btc')][string]$Group)

$ws = $PSScriptRoot
$logDir = Join-Path $ws 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "lumi_$Group.log"
$start = Get-Date

# which code produced this run? short commit, "+dirty" if the live code has uncommitted edits, "no-git"/"unknown" if git is unavailable
function Get-CodeVersion {
    try {
        $c = (git -C $ws rev-parse --short HEAD 2>$null | Out-String).Trim()
        if (-not $c) { return 'no-git' }
        $d = (git -C $ws status --porcelain -- strategies run.ps1 run_scheduled.ps1 2>$null | Out-String).Trim()
        if ($d) { return "$c+dirty" } else { return $c }
    } catch { return 'unknown' }
}
$ver = Get-CodeVersion

"=== $($start.ToString('yyyy-MM-dd HH:mm:ss')) start $Group code=$ver ===" | Out-File $log -Append -Encoding utf8
& (Join-Path $ws 'run.ps1') 'strategies\lumi_portfolio.py' 'live' $Group *>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
$code = $LASTEXITCODE
"=== $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss')) end $Group exit=$code ===" | Out-File $log -Append -Encoding utf8

$status = Join-Path $logDir 'run_status.csv'
if (-not (Test-Path $status)) { 'start,group,exit_code,seconds,code' | Out-File $status -Encoding utf8 }
"$($start.ToString('yyyy-MM-dd HH:mm:ss')),$Group,$code,$([int]((Get-Date) - $start).TotalSeconds),$ver" | Out-File $status -Append -Encoding utf8
exit $code
