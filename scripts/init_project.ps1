# SP-GQE — one-time local setup (Windows PowerShell)
# Run from repo root:  .\scripts\init_project.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path .venv)) {
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm

Write-Host ""
Write-Host "Python environment ready. Next steps:" -ForegroundColor Green
Write-Host "  1. Copy .env.example to .env"
Write-Host "  2. Install Ollama from https://ollama.com then:"
Write-Host "       ollama pull mistral"
Write-Host "       ollama pull nomic-embed-text"
Write-Host "  3. Start Neo4j:  docker compose up -d"
Write-Host "  4. Open Neo4j Browser: http://localhost:7474"
