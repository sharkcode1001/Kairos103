using Kairos103.Benchmarks.Baselines;
using Kairos103.Benchmarks.Data;
using Kairos103.Benchmarks.Metrics;
using Kairos103.Cli.Localization;
using Kairos103.Cli.Reporting;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

/// <summary>
/// Executes the complete experimental benchmarking protocol across organized datasets.
/// Formally trains both Kairos103 and baselines (Mini-Transformer and Simple LSTM) at 1.0x prior to inference evaluation.
/// </summary>
public static class RunBenchmarkCommand
{
    public static (double[] Factors, double[] KairosAcc, double[] TransAcc, double[] LstmAcc, EfficiencyTracker Tracker) Execute(
        string datasetsDir = "datasets",
        string targetDataset = "synthetic",
        double[]? customFactors = null,
        int runs = 1,
        bool fastMode = false)
    {
        string resolvedDir = LocateDatasetsDirectory(datasetsDir);
        Console.WriteLine(CliStrings.BannerTitle);
        Console.WriteLine(CliStrings.DatasetAndDir(targetDataset.ToUpperInvariant(), resolvedDir));
        if (fastMode)
        {
            Console.WriteLine(CliStrings.FastModeNotice);
        }
        Console.WriteLine();

        string subDir = targetDataset.Equals("ecg", StringComparison.OrdinalIgnoreCase) ? "ecg" : "synthetic";
        string trainFile = Path.Combine(resolvedDir, subDir, subDir == "ecg" ? "ecg_train.csv" : "warpstress_train.csv");

        if (!File.Exists(trainFile))
        {
            Console.WriteLine(CliStrings.DatasetNotFound(trainFile, Directory.GetCurrentDirectory()));
            return ([], [], [], [], new EfficiencyTracker());
        }

        Console.WriteLine(CliStrings.Step1LoadingTrain);
        var trainData = DatasetLoader.LoadCsv(trainFile);
        if (fastMode && trainData.Count > 400)
        {
            trainData = trainData.Take(400).ToList();
        }
        int maxLabel = trainData.Count == 0 ? -1 : trainData.Max(d => d.Label);
        int classCount = maxLabel + 1;
        Console.WriteLine(CliStrings.SamplesLoaded(trainData.Count, classCount));

        Console.WriteLine(CliStrings.Step2TrainingModels);
        
        Console.Write(CliStrings.TrainingKairos);
        var network = new KairosNetwork(subregionCount: 16);
        var trainer = new PairwiseTrainer(classCount, network.TotalFeatures);

        var encodedTrain = new List<(double[] Features, int Label)>(trainData.Count);
        foreach (var sample in trainData)
        {
            var (feats, _) = network.Forward(sample.Values, sample.Length);
            encodedTrain.Add((feats, sample.Label));
        }
        int kairosEpochs = fastMode ? 10 : 25;
        trainer.Train(encodedTrain, epochs: kairosEpochs, lr: 0.05, (curr, tot) =>
        {
            if (curr == 1 || curr == tot || curr % 5 == 0)
            {
                string epochStr = CliStrings.IsSpanish ? $"Época {curr}/{tot}" : $"Epoch {curr}/{tot}";
                Console.Write($"\r{CliStrings.TrainingKairos}{epochStr}");
            }
        });
        Console.WriteLine($" -> {CliStrings.Done}.");

        int tfEpochs = fastMode ? 8 : 12;
        Console.Write(CliStrings.TrainingTransformer);
        var transformer = new MiniTransformer(classCount);
        transformer.Train(trainData, epochs: tfEpochs, lr: 0.01, (curr, tot) =>
        {
            string epochStr = CliStrings.IsSpanish ? $"Época {curr}/{tot}" : $"Epoch {curr}/{tot}";
            Console.Write($"\r{CliStrings.TrainingTransformer}{epochStr}");
        });
        Console.WriteLine($" -> {CliStrings.Done}.");

        int lstmEpochs = fastMode ? 8 : 12;
        Console.Write(CliStrings.TrainingLstm);
        var lstm = new SimpleLstm(classCount);
        lstm.Train(trainData, epochs: lstmEpochs, lr: 0.01, (curr, tot) =>
        {
            string epochStr = CliStrings.IsSpanish ? $"Época {curr}/{tot}" : $"Epoch {curr}/{tot}";
            Console.Write($"\r{CliStrings.TrainingLstm}{epochStr}");
        });
        Console.WriteLine($" -> {CliStrings.Done}.");

        double[] factors = customFactors ?? [0.5, 0.8, 1.0, 1.2, 2.0];
        double[] kairosAcc = new double[factors.Length];
        double[] transformerAcc = new double[factors.Length];
        double[] lstmAcc = new double[factors.Length];

        Console.WriteLine(CliStrings.Step3EvaluatingWarp);

        // Pre-load test datasets into memory so CSV file I/O and text parsing
        // do not contaminate pure neural inference allocation metrics:
        var loadedTestSets = new (double Factor, string FactorStr, List<Kairos103.Benchmarks.Data.TimeSeriesSample> Data)?[factors.Length];
        for (int i = 0; i < factors.Length; i++)
        {
            double f = factors[i];
            string testPrefix = subDir == "ecg" ? "ecg" : "warpstress";
            string fStr = f.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture);
            string testFile = Path.Combine(resolvedDir, subDir, $"{testPrefix}_test_{fStr}x.csv");

            if (!File.Exists(testFile))
            {
                testFile = Path.Combine(resolvedDir, subDir, $"{testPrefix}_test_{f:0.0}x.csv");
            }

            if (!File.Exists(testFile))
            {
                string warnMsg = CliStrings.IsSpanish
                    ? $"      [AVISO] Archivo de test no hallado para factor {fStr}x: {testFile}"
                    : $"      [WARN] Test file not found for scale factor {fStr}x: {testFile}";
                Console.WriteLine(warnMsg);
                continue;
            }

            var testData = DatasetLoader.LoadCsv(testFile);
            if (fastMode && testData.Count > 200)
            {
                testData = testData.Take(200).ToList();
            }
            loadedTestSets[i] = (f, fStr, testData);
        }

        var effTracker = new EfficiencyTracker();
        effTracker.StartMemoryTracking();

        for (int i = 0; i < factors.Length; i++)
        {
            var item = loadedTestSets[i];
            if (item == null) continue;

            var testData = item.Value.Data;
            string fStr = item.Value.FactorStr;
            effTracker.RecordSamples(testData.Count);

            int correctKairos = 0;
            int correctTrans = 0;
            int correctLstm = 0;

            foreach (var sample in testData)
            {
                // Kairos103
                var (feats, spikes, feMacs) = network.ForwardDetailed(sample.Values, sample.Length);
                var (predKairos, accOps) = trainer.Predict(feats);
                effTracker.RecordFrontEndMacs(feMacs);
                effTracker.RecordSpikes(spikes);
                effTracker.RecordAccumulateOps(accOps);
                if (predKairos == sample.Label) correctKairos++;

                // Transformer
                var (predTrans, tMacs) = transformer.Forward(sample.Values, sample.Length);
                effTracker.RecordTransformerMacs(tMacs);
                if (predTrans == sample.Label) correctTrans++;

                // LSTM
                var (predLstm, lMacs) = lstm.Forward(sample.Values, sample.Length);
                effTracker.RecordLstmMacs(lMacs);
                if (predLstm == sample.Label) correctLstm++;
            }

            if (testData.Count == 0)
            {
                kairosAcc[i] = transformerAcc[i] = lstmAcc[i] = double.NaN;
            }
            else
            {
                kairosAcc[i] = (double)correctKairos / testData.Count;
                transformerAcc[i] = (double)correctTrans / testData.Count;
                lstmAcc[i] = (double)correctLstm / testData.Count;
            }

            Console.WriteLine($"      Factor {fStr}x -> Kairos: {kairosAcc[i] * 100.0:F1}% | Transf: {transformerAcc[i] * 100.0:F1}% | LSTM: {lstmAcc[i] * 100.0:F1}%");
        }

        effTracker.StopMemoryTracking();

        Console.WriteLine(CliStrings.Step4Summary);
        string mdTable = AsciiPlotter.GenerateMarkdownTable(targetDataset, factors, kairosAcc, transformerAcc, lstmAcc);
        Console.WriteLine(mdTable);

        Console.WriteLine(CliStrings.EfficiencyMetricsHeader);
        Console.WriteLine($"  - {CliStrings.FrontEndMacs(effTracker.TotalFrontEndMacs)}");
        Console.WriteLine($"  - {CliStrings.Spikes(effTracker.TotalSpikes)}");
        Console.WriteLine($"  - {CliStrings.ReadoutOps(effTracker.TotalAccumulateOps)}");
        Console.WriteLine($"  - {CliStrings.TransformerMacs(effTracker.TotalTransformerMacs)}");
        Console.WriteLine($"  - {CliStrings.LstmMacs(effTracker.TotalLstmMacs)}");
        Console.WriteLine($"  - {CliStrings.HeapMemory(effTracker.MemoryAllocatedBytes, effTracker.AllocatedBytesPerSample)}");

        return (factors, kairosAcc, transformerAcc, lstmAcc, effTracker);
    }

    private static string LocateDatasetsDirectory(string specifiedDir)
    {
        if (Directory.Exists(specifiedDir))
            return Path.GetFullPath(specifiedDir);

        string[] searchBases = [Directory.GetCurrentDirectory(), AppContext.BaseDirectory];
        foreach (var startDir in searchBases)
        {
            var current = new DirectoryInfo(startDir);
            while (current != null)
            {
                string candidate = Path.Combine(current.FullName, specifiedDir);
                if (Directory.Exists(candidate))
                {
                    return Path.GetFullPath(candidate);
                }
                current = current.Parent;
            }
        }

        return Path.GetFullPath(specifiedDir);
    }
}
