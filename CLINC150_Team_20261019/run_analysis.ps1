param([string]$Python = "")
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not $Python) {
    $LocalPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $LocalPython) { $Python = $LocalPython }
    else { $Python = "python" }
}
Push-Location $ProjectRoot
try {
    # Offline by default. Use data_audit.py --fetch-provenance only to pin sources.
    foreach ($AnalysisScript in @("data_audit.py", "controlled_audit.py", "error_analysis.py", "verify_analysis.py", "write_analysis_report.py")) {
        & $Python (Join-Path "src" $AnalysisScript)
        if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $AnalysisScript" }
    }
    & $Python -m unittest discover -s src -p "test_*.py" -v
    if ($LASTEXITCODE -ne 0) { throw "Analysis tests failed" }
}
finally { Pop-Location }
