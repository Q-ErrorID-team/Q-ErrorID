Set-Location "d:\ftf\mesoscopics\schools\QML2026\hackaton\Q-ErrorID-v0.61-N-qubit-support\Q-ErrorID"
.\.venv\Scripts\Activate.ps1

# Ключ задається лише в цьому вікні PowerShell, ніколи в коді/файлах.
$env:HAIQU_API_KEY = Read-Host -Prompt "Введи HAIQU_API_KEY" -AsSecureString |
    ForEach-Object { [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($_)) }

Write-Host "`n=== Крок 1: дешевий smoke-тест (512 shots, 1 validation, 1 evaluation) ===" -ForegroundColor Cyan
python scripts/demo_phonebook_correction.py `
    --shots 512 `
    --require-cloud `
    --validation-repeats 1 `
    --evaluation-repeats 1

Write-Host "`nЯкщо вище execution_mode = haiqu_cloud (без помилок) - натисни Enter для повного прогону, або Ctrl+C щоб зупинитись." -ForegroundColor Yellow
Read-Host

Write-Host "`n=== Крок 2: повний прогін (4096 shots, 2 validation, 2 evaluation) ===" -ForegroundColor Cyan
python scripts/demo_phonebook_correction.py `
    --shots 4096 `
    --require-cloud `
    --validation-repeats 2 `
    --evaluation-repeats 2

Write-Host "`nГрафік: results\haiqu\phonebook_demo_superposition.png" -ForegroundColor Green
