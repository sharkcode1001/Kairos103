using Kairos103.Cli.Commands;
using Kairos103.Cli.Localization;

// Extract language argument early if present (--lang en | --lang es)
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--lang" && i + 1 < args.Length)
    {
        CliStrings.SaveLanguagePreference(args[i + 1].ToLowerInvariant());
        break;
    }
}

// Standalone language change via CLI (e.g. kairos --lang es)
if (args.Length == 2 && args[0] == "--lang")
{
    Console.WriteLine(CliStrings.IsSpanish
        ? $"[OK] Idioma configurado y guardado permanentemente en: Español"
        : $"[OK] Language configured and saved permanently to: English");
    return;
}

if (args.Length == 0)
{
    while (true)
    {
        Console.Clear();
        Console.WriteLine("====================================================================");
        Console.WriteLine("   Kairos103: Clockless Rank-Order Neural Architecture (.NET 10)    ");
        Console.WriteLine(CliStrings.IsSpanish
            ? "                 [Idioma: Español | Language: Spanish]               "
            : "                 [Language: English | Idioma: Inglés]                ");
        Console.WriteLine("====================================================================\n");

        Console.WriteLine(CliStrings.IsSpanish ? "Seleccione una acción:" : "Select an action:");
        Console.WriteLine(CliStrings.IsSpanish
            ? "  [1] Entrenar modelo propio (Caja Negra)\n" +
              "  [2] Evaluar modelo existente (.kairos) con matriz de confusión\n" +
              "  [3] Predecir / inferir señales con un modelo guardado\n" +
              "  [4] Generar reporte oficial para Leaderboards / Internet (online-bench)\n" +
              "  [5] Ejecutar Benchmark comparativo (Kairos103 vs Transformer vs LSTM)\n" +
              "  [6] Ver ayuda de comandos CLI\n" +
              "  [7] Cambiar idioma / Change language (Español / English)\n" +
              "  [8] Pipeline completo Kaggle ECG (entrenar ensamble + generar submission.csv)\n" +
              "  [0] Salir"
            : "  [1] Train black-box model\n" +
              "  [2] Evaluate existing model (.kairos) with confusion matrix\n" +
              "  [3] Predict / infer signals with a saved model\n" +
              "  [4] Generate official Leaderboard / Internet submission (online-bench)\n" +
              "  [5] Run comparative benchmark (Kairos103 vs Transformer vs LSTM)\n" +
              "  [6] Show CLI help\n" +
              "  [7] Change language / Cambiar idioma (English / Español)\n" +
              "  [8] Full Kaggle ECG pipeline (train ensemble + generate submission.csv)\n" +
              "  [0] Exit");

        Console.Write("\n> ");
        string choice = Console.ReadLine()?.Trim() ?? "0";

        if (choice == "0")
        {
            Console.WriteLine(CliStrings.IsSpanish ? "\n¡Hasta luego!" : "\nGoodbye!");
            return;
        }

        switch (choice)
        {
            case "1":
            {
                string defData = "datasets/synthetic/warpstress_train.csv";
                Console.Write(CliStrings.IsSpanish ? $"\nRuta del CSV de entrenamiento [Enter = {defData}]: " : $"\nTraining CSV path [Enter = {defData}]: ");
                string data = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(data)) data = defData;

                string defOut = "model.kairos";
                Console.Write(CliStrings.IsSpanish ? $"Ruta de salida del modelo [Enter = {defOut}]: " : $"Model output path [Enter = {defOut}]: ");
                string outPath = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(outPath)) outPath = defOut;

                TrainCommand.Execute(data, outPath);
                break;
            }

            case "2":
            {
                string defModel = "model.kairos";
                Console.Write(CliStrings.IsSpanish ? $"\nRuta del modelo [Enter = {defModel}]: " : $"\nModel path [Enter = {defModel}]: ");
                string model = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(model)) model = defModel;

                string defData = "datasets/synthetic/warpstress_test_1.0x.csv";
                Console.Write(CliStrings.IsSpanish ? $"Ruta del CSV de prueba [Enter = {defData}]: " : $"Test CSV path [Enter = {defData}]: ");
                string data = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(data)) data = defData;

                EvaluateCommand.Execute(model, data);
                break;
            }

            case "3":
            {
                string defModel = "model.kairos";
                Console.Write(CliStrings.IsSpanish ? $"\nRuta del modelo [Enter = {defModel}]: " : $"\nModel path [Enter = {defModel}]: ");
                string model = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(model)) model = defModel;

                string defData = "datasets/synthetic/warpstress_test_1.0x.csv";
                Console.Write(CliStrings.IsSpanish ? $"Ruta del CSV de señales [Enter = {defData}]: " : $"Signal CSV path [Enter = {defData}]: ");
                string data = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(data)) data = defData;

                PredictCommand.Execute(model, data);
                break;
            }

            case "4":
            {
                OnlineBenchmarkCommand.Execute();
                break;
            }

            case "5":
            {
                Console.Write(CliStrings.IsSpanish ? "\n¿Modo rápido? (s/n) [Enter = s]: " : "\nFast mode? (y/n) [Enter = y]: ");
                string fastAns = Console.ReadLine()?.Trim().ToLowerInvariant() ?? "y";
                bool fast = fastAns != "n" && fastAns != "no";
                RunBenchmarkCommand.Execute(fastMode: fast);
                break;
            }

            case "6":
            {
                Console.WriteLine("\n" + CliStrings.AvailableUsage);
                break;
            }

            case "7":
            {
                Console.WriteLine("\nSeleccione idioma / Select language:");
                Console.WriteLine("  [1] Español (es)");
                Console.WriteLine("  [2] English (en)");
                Console.Write("> ");
                string langChoice = Console.ReadLine()?.Trim() ?? "";
                if (langChoice == "1")
                {
                    CliStrings.SaveLanguagePreference("es");
                    Console.WriteLine("\n[OK] Idioma establecido en Español.");
                }
                else if (langChoice == "2")
                {
                    CliStrings.SaveLanguagePreference("en");
                    Console.WriteLine("\n[OK] Language set to English.");
                }
                break;
            }

            case "8":
            {
                string defTrain = "datasets/nppe2_dataset/train.csv";
                string defTest = "datasets/nppe2_dataset/test.csv";
                string defSample = "datasets/nppe2_dataset/sample_submission.csv";

                Console.Write(CliStrings.IsSpanish ? $"\nCSV de entrenamiento [Enter = {defTrain}]: " : $"\nTraining CSV [Enter = {defTrain}]: ");
                string train = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(train)) train = defTrain;

                Console.Write(CliStrings.IsSpanish ? $"CSV de prueba (test) [Enter = {defTest}]: " : $"Test CSV [Enter = {defTest}]: ");
                string test = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(test)) test = defTest;

                Console.Write(CliStrings.IsSpanish ? $"sample_submission.csv [Enter = {defSample}]: " : $"sample_submission.csv [Enter = {defSample}]: ");
                string sample = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(sample)) sample = defSample;

                string defOut = "submission.csv";
                Console.Write(CliStrings.IsSpanish ? $"Salida submission [Enter = {defOut}]: " : $"Submission output [Enter = {defOut}]: ");
                string outPath = Console.ReadLine()?.Trim() ?? "";
                if (string.IsNullOrEmpty(outPath)) outPath = defOut;

                Console.Write(CliStrings.IsSpanish
                    ? "\n¿Usar ensamble de 3 semillas (42, 777, 20261010)? (s/n) [Enter = s]: "
                    : "\nUse 3-seed ensemble (42, 777, 20261010)? (y/n) [Enter = y]: ");
                string ensAns = Console.ReadLine()?.Trim().ToLowerInvariant() ?? "s";
                bool useEnsemble = ensAns != "n" && ensAns != "no";

                int[] seeds = useEnsemble ? [42, 777, 20261010] : [42];
                KagglePipelineCommand.Execute(train, test, outPath, sample, seeds: seeds);
                break;
            }

            default:
                break;
        }

        Console.WriteLine(CliStrings.IsSpanish ? "\nPresione cualquier tecla para volver al menú..." : "\nPress any key to return to menu...");
        Console.ReadKey();
    }
}

string command = args[0].ToLowerInvariant();
if (command == "--lang")
{
    command = args.Length > 2 ? args[2].ToLowerInvariant() : "help";
}

switch (command)
{
    case "benchmark":
    {
        string dataset = "synthetic";
        string dir = "datasets";
        double[]? factors = null;
        int runs = 1;
        bool fast = false;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--dataset" && i + 1 < args.Length)
            {
                dataset = args[++i];
            }
            else if (args[i] == "--datasets-dir" && i + 1 < args.Length)
            {
                dir = args[++i];
            }
            else if (args[i] == "--warp-factors" && i + 1 < args.Length)
            {
                string raw = args[++i];
                factors = raw.Split(',', StringSplitOptions.RemoveEmptyEntries)
                             .Select(s => double.Parse(s.Trim(), System.Globalization.CultureInfo.InvariantCulture))
                             .ToArray();
            }
            else if (args[i] == "--runs" && i + 1 < args.Length)
            {
                runs = int.Parse(args[++i]);
            }
            else if (args[i] == "--fast")
            {
                fast = true;
            }
            else if (args[i] == "--lang" && i + 1 < args.Length)
            {
                i++; // Already handled
            }
        }

        RunBenchmarkCommand.Execute(datasetsDir: dir, targetDataset: dataset, customFactors: factors, runs: runs, fastMode: fast);
        break;
    }

    case "jitter-sweep":
    {
        double maxJitter = 0.05;
        double step = 0.01;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--max-jitter-ms" && i + 1 < args.Length)
            {
                // If provided in ms (e.g. 50), scale down to fractional
                double val = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                maxJitter = val > 1.0 ? val / 1000.0 : val;
            }
            else if (args[i] == "--step" && i + 1 < args.Length)
            {
                double val = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                step = val > 1.0 ? val / 1000.0 : val;
            }
            else if (args[i] == "--lang" && i + 1 < args.Length)
            {
                i++; // Already handled
            }
        }

        SweepNoiseCommand.Execute(maxJitter: maxJitter, step: step);
        break;
    }

    case "export-plots":
    case "export-report":
    {
        string output = "./results";
        string dataset = "all";
        bool fast = true;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--output" && i + 1 < args.Length)
            {
                output = args[++i];
            }
            else if (args[i] == "--dataset" && i + 1 < args.Length)
            {
                dataset = args[++i];
            }
            else if (args[i] == "--full")
            {
                fast = false;
            }
            else if (args[i] == "--lang" && i + 1 < args.Length)
            {
                i++; // Already handled
            }
        }
        ExportReportCommand.Execute(outputPath: output, targetDataset: dataset, fast: fast);
        break;
    }

    case "train":
    {
        string dataPath = "";
        string outputPath = "model.kairos";
        int subregions = 16;
        int epochs = 25;
        double lr = 0.05;
        int hidden = 32;
        int seed = 20261010;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--data" && i + 1 < args.Length)
            {
                dataPath = args[++i];
            }
            else if (args[i] == "--output" && i + 1 < args.Length)
            {
                outputPath = args[++i];
            }
            else if (args[i] == "--subregions" && i + 1 < args.Length)
            {
                subregions = int.Parse(args[++i]);
            }
            else if (args[i] == "--epochs" && i + 1 < args.Length)
            {
                epochs = int.Parse(args[++i]);
            }
            else if (args[i] == "--lr" && i + 1 < args.Length)
            {
                lr = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            }
            else if (args[i] == "--hidden" && i + 1 < args.Length)
            {
                hidden = int.Parse(args[++i]);
            }
            else if (args[i] == "--seed" && i + 1 < args.Length)
            {
                seed = int.Parse(args[++i]);
            }
            else if (args[i] == "--lang" && i + 1 < args.Length)
            {
                i++;
            }
        }

        if (string.IsNullOrEmpty(dataPath))
        {
            Console.WriteLine(CliStrings.IsSpanish
                ? "[ERROR] Debe especificar el dataset con --data <ruta.csv>"
                : "[ERROR] Must specify dataset with --data <path.csv>");
            Console.WriteLine(CliStrings.AvailableUsage);
            break;
        }

        TrainCommand.Execute(dataPath, outputPath, subregions, epochs, lr, hidden, seed);
        break;
    }

    case "predict":
    {
        string modelPath = "";
        string dataPath = "";
        string? outputPath = null;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--model" && i + 1 < args.Length)
            {
                modelPath = args[++i];
            }
            else if (args[i] == "--data" && i + 1 < args.Length)
            {
                dataPath = args[++i];
            }
            else if (args[i] == "--output" && i + 1 < args.Length)
            {
                outputPath = args[++i];
            }
            else if (args[i] == "--lang" && i + 1 < args.Length)
            {
                i++;
            }
        }

        if (string.IsNullOrEmpty(modelPath) || string.IsNullOrEmpty(dataPath))
        {
            Console.WriteLine(CliStrings.IsSpanish
                ? "[ERROR] Debe especificar --model <modelo.kairos> y --data <prueba.csv>"
                : "[ERROR] Must specify --model <model.kairos> and --data <test.csv>");
            Console.WriteLine(CliStrings.AvailableUsage);
            break;
        }

        PredictCommand.Execute(modelPath, dataPath, outputPath);
        break;
    }

    case "evaluate":
    {
        string modelPath = "";
        string dataPath = "";
        string? exportJson = null;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--model" && i + 1 < args.Length)
            {
                modelPath = args[++i];
            }
            else if (args[i] == "--data" && i + 1 < args.Length)
            {
                dataPath = args[++i];
            }
            else if (args[i] == "--export-json" && i + 1 < args.Length)
            {
                exportJson = args[++i];
            }
            else if (args[i] == "--lang" && i + 1 < args.Length)
            {
                i++;
            }
        }

        if (string.IsNullOrEmpty(modelPath) || string.IsNullOrEmpty(dataPath))
        {
            Console.WriteLine(CliStrings.IsSpanish
                ? "[ERROR] Debe especificar --model <modelo.kairos> y --data <prueba.csv>"
                : "[ERROR] Must specify --model <model.kairos> and --data <test.csv>");
            Console.WriteLine(CliStrings.AvailableUsage);
            break;
        }

        EvaluateCommand.Execute(modelPath, dataPath, exportJson);
        break;
    }

    case "online-bench":
    case "bench-submit":
    {
        string dir = "datasets";
        string output = "results";

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--datasets-dir" && i + 1 < args.Length)
            {
                dir = args[++i];
            }
            else if (args[i] == "--output" && i + 1 < args.Length)
            {
                output = args[++i];
            }
            else if (args[i] == "--lang" && i + 1 < args.Length)
            {
                i++;
            }
        }

        OnlineBenchmarkCommand.Execute(dir, output);
        break;
    }

    case "train-ecg":
    {
        string trainPath = "";
        string modelPath = "results/model_ecg.kairos";
        int epochs = 35;
        double lr = 0.02;
        int hidden = 64;
        int subregions = 32;
        double valSplit = 0.2;
        int seed = 42;
        bool classWeights = true;
        bool baselines = true;
        double weightPower = 0.5;
        bool calibrateThresholds = true;
        bool useOvr = false;
        double[]? weightPowers = null;
        int cvFolds = 0;

        bool explicitSubregions = false;
        bool isKagglePreset = false;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--kaggle")
            {
                isKagglePreset = true;
                baselines = false;
                if (!explicitSubregions) subregions = 16;
            }
            else if ((args[i] == "--train" || args[i] == "--data") && i + 1 < args.Length)
                trainPath = args[++i];
            else if ((args[i] == "--model" || args[i] == "--output") && i + 1 < args.Length)
                modelPath = args[++i];
            else if (args[i] == "--epochs" && i + 1 < args.Length)
                epochs = int.Parse(args[++i]);
            else if (args[i] == "--lr" && i + 1 < args.Length)
                lr = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--hidden" && i + 1 < args.Length)
                hidden = int.Parse(args[++i]);
            else if (args[i] == "--subregions" && i + 1 < args.Length)
            {
                subregions = int.Parse(args[++i]);
                explicitSubregions = true;
            }
            else if (args[i] == "--val-split" && i + 1 < args.Length)
                valSplit = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--full-train")
                valSplit = 0.0;
            else if ((args[i] == "--cv" || args[i] == "--folds") && i + 1 < args.Length)
                cvFolds = int.Parse(args[++i]);
            else if (args[i] == "--ovr")
                useOvr = true;
            else if (args[i] == "--seed" && i + 1 < args.Length)
                seed = int.Parse(args[++i]);
            else if (args[i] == "--weight-powers" && i + 1 < args.Length)
            {
                string wpStr = args[++i];
                weightPowers = wpStr.Split(',').Select(s => double.Parse(s.Trim(), System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            }
            else if (args[i] == "--weight-power" && i + 1 < args.Length)
                weightPower = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--no-class-weights")
                classWeights = false;
            else if (args[i] == "--no-baselines")
                baselines = false;
            else if (args[i] == "--no-calibrate")
                calibrateThresholds = false;
        }

        if (isKagglePreset && !explicitSubregions)
        {
            subregions = 16;
        }

        if (string.IsNullOrEmpty(trainPath))
        {
            Console.WriteLine("[ERROR] Must specify training CSV with --train <train.csv>");
            break;
        }

        TrainEcgCommand.Execute(trainPath, modelPath, epochs, lr, hidden, subregions, valSplit, seed, classWeights, baselines, weightPower, calibrateThresholds, useOvr, weightPowers, cvFolds);
        break;
    }

    case "validate-ecg":
    {
        string modelPath = "";
        string trainPath = "";
        double valSplit = 0.2;
        int seed = 42;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--model" && i + 1 < args.Length)
                modelPath = args[++i];
            else if ((args[i] == "--train" || args[i] == "--data") && i + 1 < args.Length)
                trainPath = args[++i];
            else if (args[i] == "--val-split" && i + 1 < args.Length)
                valSplit = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
            else if (args[i] == "--seed" && i + 1 < args.Length)
                seed = int.Parse(args[++i]);
        }

        if (string.IsNullOrEmpty(modelPath) || string.IsNullOrEmpty(trainPath))
        {
            Console.WriteLine("[ERROR] Must specify --model <model_ecg.kairos> and --train <train.csv>");
            break;
        }

        ValidateEcgCommand.Execute(modelPath, trainPath, valSplit, seed);
        break;
    }

    case "predict-kaggle":
    {
        string modelPath = "";
        string testPath = "";
        string sampleSubPath = "";
        string outputPath = "submission.csv";
        double[]? customThresholds = null;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--model" && i + 1 < args.Length)
                modelPath = args[++i];
            else if ((args[i] == "--test" || args[i] == "--data") && i + 1 < args.Length)
                testPath = args[++i];
            else if ((args[i] == "--sample-submission" || args[i] == "--sample") && i + 1 < args.Length)
                sampleSubPath = args[++i];
            else if ((args[i] == "--output" || args[i] == "-o") && i + 1 < args.Length)
                outputPath = args[++i];
            else if (args[i] == "--thresholds" && i + 1 < args.Length)
            {
                string tStr = args[++i];
                customThresholds = tStr.Split(',').Select(s => double.Parse(s.Trim(), System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            }
        }

        if (string.IsNullOrEmpty(modelPath) || string.IsNullOrEmpty(testPath))
        {
            Console.WriteLine("[ERROR] Must specify --model <model.kairos> and --test <test.csv>");
            break;
        }

        PredictKaggleCommand.Execute(modelPath, testPath, outputPath, sampleSubPath, customThresholds);
        break;
    }

    case "kaggle-pipeline":
    case "run-kaggle":
    {
        KagglePipelineCommand.ExecuteFromArgs(args);
        break;
    }

    default:
    {
        Console.WriteLine(CliStrings.UnknownCommand(command));
        Console.WriteLine(CliStrings.AvailableUsage);
        break;
    }
}
