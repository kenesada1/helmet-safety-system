$ErrorActionPreference = 'Stop'

$projectRoot = 'D:\codes\helmet-safety-system'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$trainingReport = Join-Path $projectRoot 'artifacts\e7\e7_yolo11s_p2_lfr_001\e7_training_report.json'
$evaluationReport = Join-Path $projectRoot 'artifacts\e7\e7_yolo11s_p2_lfr_001\e4_e6_e7_full_val_comparison.json'
$statusPath = Join-Path $projectRoot 'artifacts\e7\e7_yolo11s_p2_lfr_001\e7_pipeline_status.json'

Set-Location -LiteralPath $projectRoot
while (-not (Test-Path -LiteralPath $trainingReport)) {
    $trainingProcesses = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and (
            $_.CommandLine -like '*scripts\train\train_e7_lfr.py*' -or
            $_.CommandLine -like '*scripts\train\resume_e7_lfr.py*'
        )
    }
    if (-not $trainingProcesses) {
        [pscustomobject]@{
            status = 'failed'
            stage = 'training'
            checked_at = [DateTime]::UtcNow.ToString('o')
            reason = 'training process exited without e7_training_report.json'
            test_used = $false
        } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
        exit 1
    }
    Start-Sleep -Seconds 30
}

if (-not (Test-Path -LiteralPath $evaluationReport)) {
    & $python -u 'scripts\evaluate\evaluate_e7_lfr.py' '--device' '0' '--batch' '2' '--workers' '0'
    if ($LASTEXITCODE -ne 0) {
        [pscustomobject]@{
            status = 'failed'
            stage = 'evaluation'
            checked_at = [DateTime]::UtcNow.ToString('o')
            exit_code = $LASTEXITCODE
            test_used = $false
        } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
        exit $LASTEXITCODE
    }
}

[pscustomobject]@{
    status = 'passed'
    stage = 'complete'
    checked_at = [DateTime]::UtcNow.ToString('o')
    training_report = $trainingReport
    evaluation_report = $evaluationReport
    test_used = $false
} | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
