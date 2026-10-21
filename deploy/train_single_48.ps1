param(
    [string]$Train = "datasets/nppe2_dataset/train.csv",
    [string]$Model = "results/single_48/model_single_48.kairos",
    [int]$Epochs = 10,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path (Split-Path $Model) | Out-Null

dotnet run --project src/Kairos103.Cli -- train-ecg `
    --train $Train `
    --model $Model `
    --epochs $Epochs `
    --lr 0.02 `
    --hidden 64 `
    --subregions 48 `
    --val-split 0.15 `
    --seed $Seed `
    --no-baselines

Write-Host "[OK] Modelo individual creado: $Model"
