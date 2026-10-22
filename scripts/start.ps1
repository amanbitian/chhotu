# Web app (http://127.0.0.1:8765) or hands-free mode:  .\start.ps1 [serve|listen|doctor]
param([ValidateSet('serve','listen','doctor')][string]$Mode = 'serve')
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $root '.venv\Scripts\python.exe') -m vox_assistant.cli $Mode
