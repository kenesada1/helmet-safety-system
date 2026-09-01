param(
    [Parameter(Mandatory = $true)]
    [int]$TrainerPid,

    [Parameter(Mandatory = $true)]
    [int]$KeepAwakePid
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$trainingReport = Join-Path $projectRoot "artifacts\e8\e8_yolo11s_p2_lfr_no_eca_001\e8_training_report.json"
$evaluationReport = Join-Path $projectRoot "artifacts\e8\e8_yolo11s_p2_lfr_no_eca_001\e4_e6_e7_e8_full_val_comparison.json"
$pipelineLog = Join-Path $projectRoot "artifacts\e8\e8_pipeline_guard.log"
$evaluationStdout = Join-Path $projectRoot "artifacts\e8\e8_evaluation_process.stdout.log"
$evaluationStderr = Join-Path $projectRoot "artifacts\e8\e8_evaluation_process.stderr.log"
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

function Write-PipelineLog {
    param([string]$Message)
    Add-Content -LiteralPath $pipelineLog -Value "$(Get-Date -Format o) $Message" -Encoding utf8
}

try {
    Write-PipelineLog "Waiting for E8 trainer PID $TrainerPid."
    Wait-Process -Id $TrainerPid
    if (-not (Test-Path -LiteralPath $trainingReport -PathType Leaf)) {
        throw "E8 trainer exited without producing e8_training_report.json"
    }
    $training = Get-Content -LiteralPath $trainingReport -Raw -Encoding utf8 | ConvertFrom-Json
    if ($training.status -ne "passed") {
        throw "E8 training report status is not passed"
    }
    if (Test-Path -LiteralPath $evaluationReport) {
        throw "Refusing to overwrite an existing E8 evaluation report"
    }
    Write-PipelineLog "Training passed; starting the fixed E4/E6/E7/E8 full-val evaluation."
    $evaluation = Start-Process -FilePath $pythonPath `
        -ArgumentList @('-u', 'scripts\evaluate\evaluate_e8_no_eca.py', '--device', '0', '--batch', '2', '--workers', '0') `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $evaluationStdout `
        -RedirectStandardError $evaluationStderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($evaluation.ExitCode -ne 0) {
        throw "E8 evaluation exited with code $($evaluation.ExitCode)"
    }
    if (-not (Test-Path -LiteralPath $evaluationReport -PathType Leaf)) {
        throw "E8 evaluation exited without producing its comparison report"
    }
    Write-PipelineLog "E8 training and evaluation both passed."
}
catch {
    Write-PipelineLog "FAILED: $($_.Exception.Message)"
}
finally {
    Stop-Process -Id $KeepAwakePid -ErrorAction SilentlyContinue
    Write-PipelineLog "Released E8 sleep-prevention process PID $KeepAwakePid."
}
