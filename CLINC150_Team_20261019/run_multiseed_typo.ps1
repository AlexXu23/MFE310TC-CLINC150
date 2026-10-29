# Reproduce the fixed-configuration 3-seed and typo workflow from the project.
# Existing complete runs are reused; training code refuses checkpoint overwrite.
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
$modelPath = Join-Path $projectPath '.models\bert-base-uncased'

function Invoke-Experiment {
    param([string[]] $Arguments)
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE" }
}

Push-Location -LiteralPath $projectPath
try {
    Invoke-Experiment @('src/test_typo_data.py')
    Invoke-Experiment @('src/test_oos_math.py')
    foreach ($trainingSeed in @(42, 43, 44)) {
        $runDirectory = "outputs/bert/seed$trainingSeed"
        if (-not (Test-Path -LiteralPath "$runDirectory/run_summary.json")) {
            Invoke-Experiment @(
                'src/train_bert.py', '--data', 'data/raw/data_full.json',
                '--output', $runDirectory, '--model-name', $modelPath,
                '--epochs', '3', '--learning-rate', '2e-5', '--train-batch-size', '16',
                '--eval-batch-size', '64', '--seed', "$trainingSeed"
            )
        }
        Invoke-Experiment @('src/validate_bert_outputs.py', '--run', $runDirectory)
        $oosDirectory = "outputs/oos/seed$trainingSeed"
        if (-not (Test-Path -LiteralPath "$oosDirectory/summary.json")) {
            Invoke-Experiment @('src/evaluate_oos.py', '--run', $runDirectory, '--output', $oosDirectory)
        }
        Invoke-Experiment @('src/verify_oos_results.py', '--run', $runDirectory, '--output', $oosDirectory, '--data', 'data/raw/data_full.json')
    }
    $typoData = 'outputs/robustness/typo_data'
    if (-not (Test-Path -LiteralPath "$typoData/summary.json")) {
        Invoke-Experiment @('src/typo_data.py', '--data', 'data/raw/data_full.json', '--output', $typoData)
    }
    $robustDirectory = 'outputs/robustness/multiseed'
    if (-not (Test-Path -LiteralPath "$robustDirectory/summary.json")) {
        Invoke-Experiment @('src/robustness.py', '--project', '.', '--typo-data', $typoData, '--output', $robustDirectory)
    }
    $summaryDirectory = 'outputs/multiseed'
    if (-not (Test-Path -LiteralPath "$summaryDirectory/summary.json")) {
        Invoke-Experiment @('src/summarize_multiseed.py', '--project', '.', '--robustness', $robustDirectory, '--output', $summaryDirectory)
    }
    Invoke-Experiment @('src/verify_robustness.py', '--project', '.', '--robustness', $robustDirectory, '--summary', $summaryDirectory, '--typo-data', $typoData)
    Invoke-Experiment @('src/write_multiseed_report.py', '--project', '.')
} finally {
    Pop-Location
}
