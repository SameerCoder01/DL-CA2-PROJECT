$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) { throw 'Run .\setup.ps1 first.' }
& .\.venv\Scripts\python.exe prepare_data.py --dataset --colorization
if ($LASTEXITCODE -ne 0) { throw 'Dataset/model downloads failed.' }
& .\.venv\Scripts\python.exe model.py --epochs 60 --batch-size 4
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect the error before retrying.' }
& .\.venv\Scripts\python.exe evaluate.py
