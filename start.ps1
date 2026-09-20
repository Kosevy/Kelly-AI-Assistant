# Kelly - Script de arranque
# Uso: .\start.ps1
# Levanta llama-server y el servidor FastAPI en ventanas separadas.

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kelly Voice Assistant - Arranque"       -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# --- 1. Verificar venv ---
$venvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $venvPath)) {
    $venvPath = Join-Path $ProjectRoot "venv"
}
if (-not (Test-Path $venvPath)) {
    Write-Error "ERROR: No se encontro el entorno virtual (.venv o venv). Crealo con: python -m venv .venv"
    exit 1
}
Write-Host "[OK] Entorno virtual encontrado: $venvPath" -ForegroundColor Green

# --- 2. Verificar llama-server ---
$llamaExe = Join-Path $ProjectRoot "llama.cpp\build\bin\Release\llama-server.exe"
if (-not (Test-Path $llamaExe)) {
    Write-Warning "ADVERTENCIA: No se encontro llama-server en: $llamaExe"
    Write-Warning "Asegurate de haber compilado llama.cpp antes de continuar."
} else {
    Write-Host "[OK] llama-server encontrado" -ForegroundColor Green
}

# --- 3. Verificar modelo GGUF ---
$modelPath = Join-Path $ProjectRoot "llama.cpp\models\qwen2.5-7b-instruct-q4_k_m.gguf"
if (-not (Test-Path $modelPath)) {
    $modelsDir = Join-Path $ProjectRoot "llama.cpp\models"
    $ggufFiles = Get-ChildItem -Path $modelsDir -Filter "*.gguf" -ErrorAction SilentlyContinue
    if ($ggufFiles) {
        $modelPath = $ggufFiles[0].FullName
        Write-Host "[INFO] Usando modelo: $($ggufFiles[0].Name)" -ForegroundColor Yellow
    } else {
        Write-Warning "ADVERTENCIA: No se encontro ningun modelo .gguf. Descarga el modelo antes de continuar."
    }
}

# --- 4. Lanzar llama-server en nueva ventana ---
if (Test-Path $llamaExe) {
    Write-Host ""
    Write-Host "[->] Lanzando llama-server (puerto 8085)..." -ForegroundColor Yellow

    # Ryzen 5 7600X: 6 nucleos fisicos. Reservamos 1 para SO + STT = --threads 5.
    # -ngl 8: offload 8 capas a GPU (~1.5 GB VRAM adicional, caben en ~2.8 GB libres).
    # Ajustar -ngl si nvidia-smi reporta OOM (bajar a 4 o 0).
    $llamaArgs = "-m `"$modelPath`" --port 8085 --host 127.0.0.1 --threads 5 --threads-batch 5 -ngl 8"

    Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/k title llama-server && `"$llamaExe`" $llamaArgs" `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Normal

    Write-Host "[OK] llama-server lanzado (threads=5, ngl=8)" -ForegroundColor Green
    Write-Host "[...] Esperando 5s para que el modelo cargue..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
} else {
    Write-Warning "[SKIP] llama-server no disponible, saltando."
}

# --- 5. Lanzar FastAPI en nueva ventana ---
Write-Host ""
Write-Host "[->] Lanzando FastAPI (puerto 8000)..." -ForegroundColor Yellow

$uvicornCmd = "`"$venvPath\Scripts\uvicorn.exe`" server:app --host 0.0.0.0 --port 8000"
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/k title Kelly-FastAPI && $uvicornCmd" `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Normal

Write-Host "[OK] FastAPI lanzado" -ForegroundColor Green

# --- 6. Mostrar URLs de acceso ---
$localIP = $null
try {
    $localIP = (Get-NetIPAddress -AddressFamily IPv4 `
        | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*" } `
        | Select-Object -First 1).IPAddress
} catch {
    # Sin IP local detectada
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  URLs de acceso:" -ForegroundColor Cyan
Write-Host "  Local:   http://localhost:8000" -ForegroundColor White
if ($localIP) {
    Write-Host "  Red LAN: http://${localIP}:8000" -ForegroundColor White
}

# Tailscale - deteccion segura
$tailscaleIP = $null
try {
    $tailscaleStatus = & tailscale status 2>$null
    if ($tailscaleStatus) {
        $match = $tailscaleStatus | Select-String -Pattern "(\d+\.\d+\.\d+\.\d+)" | Select-Object -First 1
        if ($match -and $match.Matches -and $match.Matches.Count -gt 0) {
            $tailscaleIP = $match.Matches[0].Value
        }
    }
} catch {
    # Tailscale no disponible, continuar sin error
}

if ($tailscaleIP) {
    Write-Host "  Tailscale: http://${tailscaleIP}:8000" -ForegroundColor Green
} else {
    Write-Host "  Tailscale: no detectado (esta corriendo?)" -ForegroundColor Yellow
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Kelly esta iniciando. Espera ~30s para que carguen los modelos." -ForegroundColor Yellow
Write-Host "Logs en: logs\kelly_*.log" -ForegroundColor Gray