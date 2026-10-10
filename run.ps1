# Usage: .\run.ps1 strategies\my_strategy.py [extra args]
# Runs a script with the repo's venv. Sets the CA bundle so Yahoo/pip work behind Avast HTTPS scanning.
param(
    [Parameter(Mandatory = $true)][string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ScriptArgs
)

$root = Split-Path -Parent $PSScriptRoot
$repo = Join-Path $root 'repo'
$bundle = Join-Path $repo 'ca-bundle-with-avast.pem'

if (Test-Path $bundle) {
    $env:CURL_CA_BUNDLE = $bundle
    $env:SSL_CERT_FILE = $bundle
    $env:REQUESTS_CA_BUNDLE = $bundle
}
$env:PYTHONIOENCODING = 'utf-8'

Set-Location $PSScriptRoot   # so a .env in this folder is picked up
& (Join-Path $repo '.venv\Scripts\python.exe') $Script @ScriptArgs
exit $LASTEXITCODE
