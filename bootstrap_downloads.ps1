# Run this once from your own PowerShell when the Work session cannot access the network/Downloads.
# Clones into Downloads, installs signed Python 3.12 if needed, installs dependencies,
# starts the local app, then downloads the data and trains with live console output.
$ErrorActionPreference = 'Stop'
$taskDestination = Join-Path $env:USERPROFILE 'Downloads\DL-CA2-PROJECT'
$taskPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
if (-not (Test-Path -LiteralPath $taskDestination)) {
    git clone https://github.com/SameerCoder01/DL-CA2-PROJECT.git $taskDestination
    if ($LASTEXITCODE -ne 0) { throw 'Git clone failed. Check that Git and internet access are available.' }
} else {
    $taskExistingRemote = git -C $taskDestination remote get-url origin
    if ($LASTEXITCODE -ne 0 -or $taskExistingRemote -notmatch '^https://github\.com/SameerCoder01/DL-CA2-PROJECT(?:\.git)?/?$') {
        throw 'Downloads\DL-CA2-PROJECT already exists and is not this repository. No files were overwritten.'
    }
    git -C $taskDestination pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'Pull failed. Preserve and inspect any local changes before continuing.' }
}
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskInstaller = Join-Path $env:TEMP 'DL-CA2-python-3.12.10-amd64.exe'
    Invoke-WebRequest 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $taskInstaller
    $taskSignature = Get-AuthenticodeSignature -LiteralPath $taskInstaller
    if ($taskSignature.Status -ne 'Valid' -or $taskSignature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
        throw 'Python installer signature verification failed.'
    }
    $taskInstall = Start-Process -FilePath $taskInstaller -ArgumentList @('/quiet','InstallAllUsers=0','Include_pip=1','Include_test=0','Include_launcher=1','InstallLauncherAllUsers=0','PrependPath=0') -Wait -PassThru -WindowStyle Hidden
    if ($taskInstall.ExitCode -notin @(0,3010) -or -not (Test-Path -LiteralPath $taskPython)) {
        throw 'Python installation failed. Install Python 3.12 x64 from python.org, then run this script again.'
    }
}
Set-Location -LiteralPath $taskDestination
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & $taskPython -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& .\setup.ps1
if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed.' }
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Project tests failed; training was not started.' }
& .\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU unavailable. Check PyTorch and driver setup before a long training run.'"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed; training was not started.' }
New-Item -ItemType Directory -Force outputs | Out-Null
$taskAppExecutable = Join-Path $taskDestination '.venv\Scripts\python.exe'
$taskAppProcess = Start-Process -FilePath $taskAppExecutable -WorkingDirectory $taskDestination -ArgumentList @('-m','streamlit','run','App.py','--server.headless=true') -RedirectStandardOutput 'outputs\app.log' -RedirectStandardError 'outputs\app-errors.log' -WindowStyle Hidden -PassThru
Write-Host "App process: $($taskAppProcess.Id). Open http://localhost:8501"
Write-Host 'Training will now run in this console. Keep it open; completion time depends on the GPU.'
& .\train.ps1
if ($LASTEXITCODE -ne 0) { throw 'Training or evaluation failed. Read the error above.' }
Write-Host 'Training and evaluation complete. Refresh the app and select Trained U-Net.'
