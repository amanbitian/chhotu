# Cloned-voice TTS service. Runs inside the Audio Clone virtualenv, which owns the ROCm PyTorch + VoxCPM2 install.
param([switch]$NoPreload)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$clone = if ($env:VOXA_AUDIO_CLONE_ROOT) { $env:VOXA_AUDIO_CLONE_ROOT } else { 'F:\Projects\Audio Clone' }
$py = Join-Path $clone '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { throw "Audio Clone environment not found at $clone" }
$env:PYTHONPATH = Join-Path $root 'src'
$env:PYTHONUTF8 = '1'
if (-not $env:VOXA_DATA_DIR) { $env:VOXA_DATA_DIR = Join-Path $root 'data' }
$extra = @(); if (-not $NoPreload) { $extra += '--preload' }
# uvicorn logs to stderr; with 'Stop', PowerShell 5.1 would abort on its first line whenever output is redirected.
$ErrorActionPreference = 'Continue'
& $py -m vox_assistant.tts_service @extra
exit $LASTEXITCODE
