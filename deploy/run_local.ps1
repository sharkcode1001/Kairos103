<#
.SYNOPSIS
    Ejecuta el pipeline completo de Kaggle (NPPE2-T2-26 ECG) de forma local, 100% .NET (sin Python).

.DESCRIPTION
    Replica lo que hacia el notebook de Kaggle, pero usando los CSV que tengas en tu PC:
      1. Compila Kairos103.Cli en Release.
      2. Entrena N modelos OvR (multi-semilla) con el 100% del train (--full-train).
      3. Genera submission.csv por inferencia de ensamble con thresholds calibrados.
      4. Verifica la integridad del submission generado.

.PARAMETER TrainPath
    Ruta al train.csv (relativa a la raiz del repo o absoluta).

.PARAMETER TestPath
    Ruta al test.csv.

.PARAMETER SampleSubmission
    Ruta al sample_submission.csv (opcional, se usa para alinear IDs).

.PARAMETER Output
    Ruta del submission.csv de salida.

.PARAMETER ModelDir
    Carpeta donde se guardan los modelos .kairos.

.PARAMETER Seeds
    Semillas de los modelos del ensamble (por defecto 42,777,20261010).

.PARAMETER Thresholds
    Thresholds de decision por clase: t0,t1,t2,t3.

.EXAMPLE
    pwsh -File deploy/run_local.ps1

.EXAMPLE
    pwsh -File deploy/run_local.ps1 -Seeds 42,777,20261010 -Epochs 28 -Subregions 40

.NOTES
    Ejecutar desde la raiz del repositorio:  .\deploy\run_local.ps1
    Si la politica de ejecucion bloquea el script:
        powershell -ExecutionPolicy Bypass -File .\deploy\run_local.ps1
#>
[CmdletBinding()]
param(
    [string]$TrainPath = "datasets/nppe2_dataset/train.csv",
    [string]$TestPath = "datasets/nppe2_dataset/test.csv",
    [string]$SampleSubmission = "datasets/nppe2_dataset/sample_submission.csv",
    [string]$Output = "submission.csv",
    [string]$ModelDir = "results",
    [int]$Subregions = 40,
    [int]$Hidden = 64,
    [int]$Epochs = 28,
    [double]$Lr = 0.02,
    [string]$WeightPowers = "0.35,0.70,0.50,0.85",
    [int[]]$Seeds = @(42, 777, 20261010),
    [string]$Thresholds = "1.00,0.85,1.25,0.85",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$invariant = [System.Globalization.CultureInfo]::InvariantCulture

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $repoRoot $Path)
}

function Invoke-Step {
    param([string]$Label, [scriptblock]$Action)
    Write-Host ""
    Write-Host ("=" * 74)
    Write-Host "  $Label"
    Write-Host ("=" * 74)
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "El paso '$Label' termino con codigo de salida $LASTEXITCODE."
    }
}

$cliProject = Join-Path $repoRoot "src/Kairos103.Cli/Kairos103.Cli.csproj"
$trainFull = Resolve-RepoPath $TrainPath
$testFull = Resolve-RepoPath $TestPath
$sampleFull = Resolve-RepoPath $SampleSubmission
$outputFull = Resolve-RepoPath $Output
$modelDirFull = Resolve-RepoPath $ModelDir
$lrStr = $Lr.ToString($invariant)

foreach ($required in @($trainFull, $testFull)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "No se encontro el archivo requerido: $required"
    }
}

if (-not (Test-Path -LiteralPath $modelDirFull)) {
    New-Item -ItemType Directory -Path $modelDirFull -Force | Out-Null
}

$outputDir = Split-Path -Parent $outputFull
if (-not [string]::IsNullOrEmpty($outputDir) -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "  KAIROS103 - PIPELINE KAGGLE LOCAL (.NET 10, sin Python)"
Write-Host "======================================================================"
Write-Host "  Train:        $trainFull"
Write-Host "  Test:         $testFull"
Write-Host "  Submission:   $outputFull"
Write-Host "  Modelos:      $modelDirFull"
Write-Host "  Semillas:     $($Seeds -join ', ')"
Write-Host "  Subregiones:  $Subregions ($($Subregions * 4) neuronas LIF)"
Write-Host "  Hidden:       $Hidden | Epochs: $Epochs | Lr: $lrStr"
Write-Host "  WeightPowers: $WeightPowers"
Write-Host "  Thresholds:   $Thresholds"
Write-Host "======================================================================"

Invoke-Step "[1/3] Compilando Kairos103.Cli ($Configuration)" {
    dotnet build $cliProject -c $Configuration
}

$modelPaths = @()
$modelIndex = 0
$trainBase = [System.IO.Path]::GetFileNameWithoutExtension($trainFull)
foreach ($seed in $Seeds) {
    $modelIndex++
    $modelPath = Join-Path $modelDirFull "model_local_${trainBase}_$seed.kairos"
    $label = "[2/3] Entrenando Modelo $modelIndex/$($Seeds.Count) (Semilla $seed, $Subregions subregiones, 100% data)"
    $mPath = $modelPath
    $s = $seed
    Invoke-Step $label {
        dotnet run --project $cliProject -c $Configuration --no-build -- `
            train-ecg --train $trainFull --model $mPath `
            --subregions $Subregions --hidden $Hidden --epochs $Epochs --lr $lrStr `
            --seed $s --weight-powers $WeightPowers --ovr --full-train --no-baselines
    }
    if (-not (Test-Path -LiteralPath $modelPath)) {
        throw "El modelo esperado no se genero: $modelPath"
    }
    $modelPaths += $modelPath
}

$modelsArg = ($modelPaths -join ",")
Invoke-Step "[3/3] Generando Inferencia de Ensamble de $($modelPaths.Count) Modelos" {
    $predictArgs = @(
        "run", "--project", $cliProject, "-c", $Configuration, "--no-build", "--",
        "predict-kaggle", "--model", $modelsArg, "--test", $testFull,
        "--output", $outputFull, "--thresholds", $Thresholds
    )
    if (-not [string]::IsNullOrEmpty($SampleSubmission) -and (Test-Path -LiteralPath $sampleFull)) {
        $predictArgs += @("--sample-submission", $sampleFull)
    }
    dotnet @predictArgs
}

if (-not (Test-Path -LiteralPath $outputFull)) {
    throw "No se genero el submission: $outputFull"
}

$expectedRows = $null
if (Test-Path -LiteralPath $sampleFull) {
    $expectedRows = (Get-Content -LiteralPath $sampleFull).Count - 1
}
$actualRows = (Get-Content -LiteralPath $outputFull).Count - 1

Write-Host ""
Write-Host "======================================================================"
Write-Host "  PIPELINE LOCAL COMPLETADO"
Write-Host "======================================================================"
Write-Host "  Submission:    $outputFull"
Write-Host "  Filas:         $actualRows"
if ($null -ne $expectedRows) {
    Write-Host "  Filas sample:  $expectedRows"
}
Write-Host ""
Write-Host "  Sube '$([System.IO.Path]::GetFileName($outputFull))' a Kaggle para evaluarlo."
Write-Host "======================================================================"
