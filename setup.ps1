# 다른 PC에서 처음 한 번만 실행 (Windows PowerShell)
# 사용법:  PowerShell에서  ./setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/3] 가상환경(.venv) 생성..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }

Write-Host "[2/3] 패키지 설치..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "[3/3] 완료! 앱 실행하려면:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\streamlit.exe run app.py" -ForegroundColor Yellow
