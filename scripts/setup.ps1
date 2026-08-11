$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"

if (-not (Test-Path $VenvPath)) {
    $Created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($Version in @("3.12", "3.11")) {
            & py "-$Version" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                & py "-$Version" -m venv $VenvPath
                if ($LASTEXITCODE -eq 0) {
                    $Created = $true
                    break
                }
            }
        }
    }
    if (-not $Created -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 veya 3.12 gerekli."
        }
        & python -m venv $VenvPath
        $Created = $LASTEXITCODE -eq 0
    }
    if (-not $Created) {
        throw "Python 3.11 veya 3.12 bulunamadi. Python kurup scripti yeniden calistirin."
    }
}

$Python = Join-Path $VenvPath "Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -e "$ProjectRoot[dev]"
& $Python -m market_moe.cli doctor

Write-Host "MarketMoE kurulumu tamamlandi."
Write-Host "Baslatmak icin: .\.venv\Scripts\market-moe.exe serve"
