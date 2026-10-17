using System.Globalization;
using Kairos103.Core.Data;

namespace Kairos103.Cli.Commands;

/// <summary>
/// Orquestador del pipeline completo de la competicion Kaggle ECG (NPPE2-T2-26):
/// entrena un ensamble multi-semilla en modo OvR (100% de los datos) y genera el submission.csv.
/// Todo en .NET, sin dependencias de Python ni de rutas especificas de Kaggle.
/// </summary>
public static class KagglePipelineCommand
{
    public static void Execute(
        string trainPath,
        string testPath,
        string outputPath = "submission.csv",
        string? sampleSubmissionPath = null,
        string modelDir = "results",
        int subregions = 40,
        int hidden = 64,
        int epochs = 28,
        double lr = 0.02,
        string weightPowers = "0.35,0.70,0.50,0.85",
        int[]? seeds = null,
        string? thresholds = null,
        int cvFolds = 5)
    {
        Console.WriteLine("\n==========================================================================");
        Console.WriteLine("     KAIROS103 - PIPELINE KAGGLE ECG (LOCAL, 100% .NET)                   ");
        Console.WriteLine("==========================================================================");

        if (string.IsNullOrWhiteSpace(trainPath) || !File.Exists(trainPath))
        {
            Console.WriteLine($"[ERROR] No se encontro el archivo de entrenamiento: {trainPath}");
            return;
        }
        if (string.IsNullOrWhiteSpace(testPath) || !File.Exists(testPath))
        {
            Console.WriteLine($"[ERROR] No se encontro el archivo de prueba: {testPath}");
            return;
        }

        int[] seedList = (seeds != null && seeds.Length > 0) ? seeds : [42, 777, 20261010];
        double[] weightPowersArr = ParseVector(weightPowers);
        double[]? thresholdsArr = !string.IsNullOrWhiteSpace(thresholds) ? ParseVector(thresholds) : null;

        if (weightPowersArr.Length != 4)
        {
            Console.WriteLine("[ERROR] --weight-powers debe contener 4 valores (uno por clase).");
            return;
        }
        if (thresholdsArr != null && thresholdsArr.Length != 4)
        {
            Console.WriteLine("[ERROR] --thresholds debe contener 4 valores (uno por clase).");
            return;
        }

        Directory.CreateDirectory(modelDir);
        string? outputDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDir)) Directory.CreateDirectory(outputDir);

        Console.WriteLine($"  Train:        {trainPath}");
        Console.WriteLine($"  Test:         {testPath}");
        Console.WriteLine($"  Submission:   {outputPath}");
        Console.WriteLine($"  Modelos:      {modelDir}");
        Console.WriteLine(cvFolds >= 2 
            ? $"  Modo:         {cvFolds}-Fold Cross-Validation (OOF Calibration)" 
            : $"  Modo:         Full-Train Ensamble ({seedList.Length} semillas)");
        Console.WriteLine($"  Subregiones:  {subregions} ({subregions * 4} neuronas LIF)");
        Console.WriteLine($"  Hidden:       {hidden} | Epochs: {epochs} | Lr: {lr.ToString(CultureInfo.InvariantCulture)}");
        Console.WriteLine($"  WeightPowers: {weightPowers}");
        Console.WriteLine(thresholdsArr != null 
            ? $"  Thresholds:   {thresholds} (manual override)" 
            : $"  Thresholds:   Auto OOF Calibrated (Matematico)");
        Console.WriteLine("==========================================================================");

        string baseName = Path.GetFileNameWithoutExtension(trainPath);
        Console.WriteLine($"\nLoading training dataset: {trainPath}...");
        var allSamples = KaggleEcgDataset.LoadTrain(trainPath);
        Console.WriteLine($"Loaded {allSamples.Count:N0} samples.\n");

        List<string> modelPaths;

        if (cvFolds >= 2)
        {
            string baseModelPath = Path.Combine(modelDir, $"model_{baseName}.kairos");
            modelPaths = TrainEcgCommand.Execute(
                trainPath: trainPath,
                modelPath: baseModelPath,
                epochs: epochs,
                lr: lr,
                hiddenDim: hidden,
                subregions: subregions,
                valSplit: 0.0,
                seed: seedList[0],
                useClassWeights: true,
                runBaselines: false,
                weightPower: 0.5,
                calibrateThresholds: true,
                useOvr: true,
                weightPowers: weightPowersArr,
                cvFolds: cvFolds,
                preloadedSamples: allSamples,
                runDiagnostics: false);
        }
        else
        {
            modelPaths = new List<string>(seedList.Length);
            for (int i = 0; i < seedList.Length; i++)
            {
                int seed = seedList[i];
                string modelPath = Path.Combine(modelDir, $"model_local_{baseName}_{seed}.kairos");

                Console.WriteLine("\n==========================================================================");
                Console.WriteLine($"  Entrenando Modelo {i + 1}/{seedList.Length} (Semilla {seed}, {subregions} subregiones, 100% data)");
                Console.WriteLine("==========================================================================");

                TrainEcgCommand.Execute(
                    trainPath: trainPath,
                    modelPath: modelPath,
                    epochs: epochs,
                    lr: lr,
                    hiddenDim: hidden,
                    subregions: subregions,
                    valSplit: 0.0,
                    seed: seed,
                    useClassWeights: true,
                    runBaselines: false,
                    weightPower: 0.5,
                    calibrateThresholds: false,
                    useOvr: true,
                    weightPowers: weightPowersArr,
                    cvFolds: 0,
                    preloadedSamples: allSamples,
                    runDiagnostics: false);

                if (!File.Exists(modelPath))
                {
                    Console.WriteLine($"[ERROR] El modelo esperado no se genero: {modelPath}");
                    return;
                }
                modelPaths.Add(modelPath);
            }
        }

        Console.WriteLine("\n==========================================================================");
        Console.WriteLine($"  Generando Inferencia de Ensamble de {modelPaths.Count} Modelos");
        Console.WriteLine("==========================================================================");

        PredictKaggleCommand.Execute(
            modelPath: string.Join(",", modelPaths),
            testPath: testPath,
            outputPath: outputPath,
            sampleSubmissionPath: sampleSubmissionPath,
            customThresholds: thresholdsArr);
    }

    private static double[] ParseVector(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return [];
        return raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                  .Select(s => double.Parse(s, CultureInfo.InvariantCulture))
                  .ToArray();
    }

    private static int[] ParseSeeds(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return [];
        return raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                  .Select(s => int.Parse(s, CultureInfo.InvariantCulture))
                  .ToArray();
    }

    /// <summary>
    /// Punto de entrada que parsea los argumentos de la CLI para el subcomando "kaggle-pipeline".
    /// </summary>
    public static void ExecuteFromArgs(string[] args)
    {
        string trainPath = "datasets/nppe2_dataset/train.csv";
        string testPath = "datasets/nppe2_dataset/test.csv";
        string samplePath = "datasets/nppe2_dataset/sample_submission.csv";
        string outputPath = "submission.csv";
        string modelDir = "results";
        int subregions = 40;
        int hidden = 64;
        int epochs = 28;
        double lr = 0.02;
        string weightPowers = "0.35,0.70,0.50,0.85";
        string? thresholds = null;
        int[]? seeds = null;
        int cvFolds = 5;

        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--train":
                case "--data":
                    if (i + 1 < args.Length) trainPath = args[++i];
                    break;
                case "--test":
                    if (i + 1 < args.Length) testPath = args[++i];
                    break;
                case "--sample-submission":
                case "--sample":
                    if (i + 1 < args.Length) samplePath = args[++i];
                    break;
                case "--output":
                case "-o":
                    if (i + 1 < args.Length) outputPath = args[++i];
                    break;
                case "--model-dir":
                    if (i + 1 < args.Length) modelDir = args[++i];
                    break;
                case "--subregions":
                    if (i + 1 < args.Length) subregions = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--hidden":
                    if (i + 1 < args.Length) hidden = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--epochs":
                    if (i + 1 < args.Length) epochs = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--lr":
                    if (i + 1 < args.Length) lr = double.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--weight-powers":
                    if (i + 1 < args.Length) weightPowers = args[++i];
                    break;
                case "--thresholds":
                    if (i + 1 < args.Length) thresholds = args[++i];
                    break;
                case "--cv-folds":
                case "--kfold":
                case "--folds":
                    if (i + 1 < args.Length) cvFolds = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--seeds":
                    if (i + 1 < args.Length) seeds = ParseSeeds(args[++i]);
                    break;
                case "--lang":
                    if (i + 1 < args.Length) i++; // handled globally
                    break;
            }
        }

        Execute(trainPath, testPath, outputPath, samplePath, modelDir, subregions, hidden, epochs, lr, weightPowers, seeds, thresholds, cvFolds);
    }
}
