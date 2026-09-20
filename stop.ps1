# Kelly - Script de parada
# Uso: .\stop.ps1
# Detiene llama-server y el proceso uvicorn de Kelly.

Write-Host "Deteniendo Kelly..." -ForegroundColor Yellow

$stopped = @()

# Matar llama-server
$llamaProcs = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($llamaProcs) {
    $llamaProcs | Stop-Process -Force
    $stopped += "llama-server ($($llamaProcs.Count) proceso(s))"
}

# Matar uvicorn (buscar por linea de comandos)
$uvicornProcs = Get-WmiObject Win32_Process `
    | Where-Object { $_.CommandLine -like "*uvicorn*server:app*" }
if ($uvicornProcs) {
    $uvicornProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    $stopped += "uvicorn ($($uvicornProcs.Count) proceso(s))"
}

# Matar python procesos con server.py (fallback)
$pyProcs = Get-WmiObject Win32_Process `
    | Where-Object { $_.CommandLine -like "*server.py*" -or $_.CommandLine -like "*kelly*" }
if ($pyProcs) {
    $pyProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    $stopped += "python/kelly ($($pyProcs.Count) proceso(s))"
}

if ($stopped.Count -gt 0) {
    Write-Host "[OK] Detenidos: $($stopped -join ', ')" -ForegroundColor Green
} else {
    Write-Host "[INFO] No se encontraron procesos activos de Kelly." -ForegroundColor Gray
}