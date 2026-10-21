param(
    [string]$Train = "datasets/nppe2_dataset/train.csv",
    [string]$Test = "datasets/nppe2_dataset/test.csv",
    [string]$SampleSubmission = "datasets/nppe2_dataset/sample_submission.csv",
    [string]$OutputDir = "results/best_ensemble",
    [int]$Epochs = 28,
    [int]$Subregions = 16,
    [int]$Hidden = 64,
    [double]$LearningRate = 0.02
)

$ErrorActionPreference = "Stop"
$project = "src/Kairos103.Cli"
$models = @()

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "[1/2] Entrenando 3 modelos OvR con validacion del 15% y thresholds calibrados..."
foreach ($seed in @(42, 777, 20261010)) {
    $model = Join-Path $OutputDir ("record_ovr_seed_{0}.kairos" -f $seed)
    if (Test-Path $model) {
        Write-Host "  [SKIP] Ya existe: $model"
    }
    else {
        dotnet run --project $project -- train-ecg `
            --train $Train --model $model --epochs $Epochs --lr $LearningRate `
            --hidden $Hidden --subregions $Subregions --val-split 0.15 --seed $seed `
            --ovr --weight-powers 0.35,0.70,0.50,0.85 --no-baselines
    }
    $models += $model
}
$modelArg = $models -join ","
$submission = Join-Path $OutputDir "submission_record_ovr_3seed.csv"

Write-Host "[2/2] Generando submission con el ensamble OvR de 3 semillas..."
$predictArgs = @("predict-kaggle", "--model", $modelArg, "--test", $Test, "--output", $submission)
if ($SampleSubmission -and (Test-Path $SampleSubmission)) {
    $predictArgs += @("--sample-submission", $SampleSubmission)
}
dotnet run --project $project -- @predictArgs

Write-Host "[OK] Submission creada: $submission"
