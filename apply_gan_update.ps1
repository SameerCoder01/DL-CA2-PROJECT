# Run this prepared file from the Work project copy in your own PowerShell.
# Reuses the installed Downloads environment; never deletes data or old weights.
$ErrorActionPreference = 'Stop'
$taskSource = $PSScriptRoot
$taskDestination = Join-Path $env:USERPROFILE 'Downloads\DL-CA2-PROJECT'
$taskExpectedHead = '503a06d1b76b3c93a89869388ee694bec02d83a7'
$taskFiles = @('App.py','model.py','evaluate.py','prepare_data.py','README.md','VIVA_GUIDE.md','VALIDATION.md',
    'tests\test_pipeline.py','notebooks\Train_on_Colab.ipynb','bootstrap_downloads.ps1',
    'apply_gan_update.ps1')

if (-not (Test-Path -LiteralPath (Join-Path $taskDestination '.git'))) {
    throw 'The Downloads clone was not found. No files were changed.'
}
if ([IO.Path]::GetFullPath($taskSource) -eq [IO.Path]::GetFullPath($taskDestination)) {
    throw 'Run the prepared update from the Work project copy, not from inside Downloads.'
}
$taskRemote = git -C $taskDestination remote get-url origin
if ($LASTEXITCODE -ne 0 -or $taskRemote -notmatch '^https://github\.com/SameerCoder01/DL-CA2-PROJECT(?:\.git)?/?$') {
    throw 'The Downloads folder is not the expected repository.'
}
git -C $taskDestination diff --quiet
if ($LASTEXITCODE -ne 0) { throw 'Tracked local edits exist. Preserve them before applying this update.' }
git -C $taskDestination diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Staged changes exist. Preserve them before applying this update.' }
git -C $taskDestination pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git pull failed. No project files were overwritten.' }
$taskHead = git -C $taskDestination rev-parse HEAD
if ($taskHead.Trim() -ne $taskExpectedHead) {
    throw 'The repository has changed since this update was prepared. Stop and review it before replacing files.'
}
foreach ($taskFile in $taskFiles) {
    $taskOriginal = Join-Path $taskSource $taskFile
    $taskTarget = Join-Path $taskDestination $taskFile
    if (-not (Test-Path -LiteralPath $taskOriginal)) { throw "Update source is missing: $taskFile" }
    $taskTracked = git -C $taskDestination ls-files -- $taskFile
    if (-not $taskTracked -and (Test-Path -LiteralPath $taskTarget)) {
        throw "Untracked file would be replaced: $taskFile. Preserve it first."
    }
}
foreach ($taskFile in $taskFiles) {
    Copy-Item -LiteralPath (Join-Path $taskSource $taskFile) -Destination (Join-Path $taskDestination $taskFile)
}
Set-Location -LiteralPath $taskDestination
$taskPython = Join-Path $taskDestination '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Existing project environment is missing.' }
& $taskPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'GAN tests failed. Training and push were not started.' }
git add -- $taskFiles
if ($LASTEXITCODE -ne 0) { throw 'Could not stage the GAN update.' }
git commit -m 'Replace supervised restoration with explainable Pix2Pix GAN'
if ($LASTEXITCODE -ne 0) { throw 'Git commit failed. Updated files remain available locally.' }
git push origin main
if ($LASTEXITCODE -ne 0) { Write-Warning 'GitHub push failed. GAN commit is saved locally; training will continue. Retry git push after resolving GitHub login.' }
& $taskPython -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print('GAN GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
& $taskPython prepare_data.py --dataset
if ($LASTEXITCODE -ne 0) { throw 'Dataset download failed. The code update is already saved.' }
& $taskPython prepare_data.py --colorization
if ($LASTEXITCODE -ne 0) { Write-Warning 'Optional colorization assets unavailable. GAN restoration training will continue.' }
& $taskPython model.py --epochs 60 --samples 4000 --batch-size 4
if ($LASTEXITCODE -ne 0) { throw 'GAN training stopped. Read the error above; existing checkpoints are preserved.' }
& $taskPython evaluate.py
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed. Training checkpoints are preserved.' }
Write-Host 'GAN training and evaluation complete. Starting the Streamlit app.'
& $taskPython -m streamlit run App.py
