param(
    [string]$Train = "datasets/nppe2_dataset/train.csv",
    [string]$Test = "datasets/nppe2_dataset/test.csv",
    [string]$SampleSubmission = "datasets/nppe2_dataset/sample_submission.csv",
    [string]$OutputDir = "results/master_ensemble_40",
    [int]$Epochs = 10,
    [int]$Subregions = 40,
    [int]$Hidden = 64,
    [double]$LearningRate = 0.02
)

$ErrorActionPreference = "Stop"
$project = "src/Kairos103.Cli"
$foldBase = Join-Path $OutputDir "model_40fold.kairos"
$models = @()
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "[1/3] Entrenando 5 folds con $Subregions subregiones..."
$foldExpected = 1..5 | ForEach-Object { Join-Path $OutputDir ("model_40fold_fold{0}.kairos" -f $_) }
$missingFold = $foldExpected | Where-Object { -not (Test-Path $_) }
if ($missingFold.Count -gt 0) {
    dotnet run --project $project -- train-ecg `
        --train $Train --model $foldBase --epochs $Epochs --lr $LearningRate `
        --hidden $Hidden --subregions $Subregions --cv 5 --seed 42 --no-baselines
}
$models += $foldExpected | Where-Object { Test-Path $_ }

Write-Host "[2/3] Entrenando 3 modelos full-train con $Subregions subregiones..."
foreach ($seed in @(42, 777, 20261010)) {
    $model = Join-Path $OutputDir ("model_40full_seed_{0}.kairos" -f $seed)
    if (-not (Test-Path $model)) {
        dotnet run --project $project -- train-ecg `
            --train $Train --model $model --epochs $Epochs --lr $LearningRate `
            --hidden $Hidden --subregions $Subregions --full-train --seed $seed `
            --no-baselines
    }
    $models += $model
}

if ($models.Count -ne 8) {
    throw "Se esperaban 8 modelos, pero solo hay $($models.Count). Revisa el entrenamiento de folds."
}

$submission = Join-Path $OutputDir "submission_master_ensemble_40.csv"
$modelArg = $models -join ","
Write-Host "[3/3] Generando submission con los 8 modelos..."
$predictArgs = @("predict-kaggle", "--model", $modelArg, "--test", $Test, "--output", $submission)
if ($SampleSubmission -and (Test-Path $SampleSubmission)) {
    $predictArgs += @("--sample-submission", $SampleSubmission)
}
dotnet run --project $project -- @predictArgs
Write-Host "[OK] Submission creada: $submission"
