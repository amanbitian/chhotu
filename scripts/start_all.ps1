# Everything, each in its own window: the voice LLM, the cloned-voice service, then the web app or hands-free mode.
#   .\scripts\start_all.ps1            # web app at http://127.0.0.1:8765
#   .\scripts\start_all.ps1 listen     # hands-free
param([ValidateSet('serve','listen')][string]$Mode = 'serve')
$here = $PSScriptRoot
$root = Split-Path -Parent $here
$py = Join-Path $root '.venv\Scripts\python.exe'

function Wait-Url($url, $what, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try { Invoke-RestMethod $url -TimeoutSec 2 | Out-Null; Write-Host "  $what is up"; return $true } catch { Start-Sleep -Seconds 1 }
    }
    Write-Host "  $what did not come up within $seconds s; see its window" -ForegroundColor Yellow
    return $false
}

Write-Host "Starting the voice LLM and the cloned-voice service..."
try { Invoke-RestMethod http://127.0.0.1:8083/v1/models -TimeoutSec 2 | Out-Null; Write-Host "  voice LLM already running" }
catch { Start-Process powershell -ArgumentList '-NoExit', '-File', (Join-Path $here 'start_llm.ps1') | Out-Null }
try { Invoke-RestMethod http://127.0.0.1:8870/health -TimeoutSec 2 | Out-Null; Write-Host "  voice service already running" }
catch { Start-Process powershell -ArgumentList '-NoExit', '-File', (Join-Path $here 'start_tts.ps1') | Out-Null }
Wait-Url 'http://127.0.0.1:8083/v1/models' 'voice LLM' 120 | Out-Null
Wait-Url 'http://127.0.0.1:8870/health' 'voice service (loads and warms the model)' 180 | Out-Null
& $py -m vox_assistant.cli $Mode
