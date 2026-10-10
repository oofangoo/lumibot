# Starts the read-only dashboard (if not already running) and opens it in the default browser.
$url = 'http://127.0.0.1:8765'
$up = $false
try { Invoke-WebRequest -UseBasicParsing "$url/api/doc?name=diary" -TimeoutSec 2 | Out-Null; $up = $true } catch {}
if (-not $up) {
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'run.ps1'), 'dashboard\server.py'
    Start-Sleep 6
}
Start-Process $url
