# Only downloads the official dataset into this project folder.
# Run in your own PowerShell if the Work terminal's network is restricted.
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Python 3.12 executable was not found.' }
& $taskPython (Join-Path $PSScriptRoot 'prepare_data.py') --dataset
if ($LASTEXITCODE -ne 0) { throw 'Dataset download failed. Read the output above.' }
Write-Host 'Dataset is ready in the Work project folder. No training was started by this download script.'
