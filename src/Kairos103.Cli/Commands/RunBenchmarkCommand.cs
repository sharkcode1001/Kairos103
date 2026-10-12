using Kairos103.Benchmarks.Baselines;
using Kairos103.Benchmarks.Data;
using Kairos103.Benchmarks.Metrics;
using Kairos103.Cli.Reporting;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

/// <summary>
/// Ejecuta el protocolo experimental completo de benchmarking sobre los datasets organizados.
/// Entrena legítimamente tanto Kairos103 como los baselines (Mini-Transformer y Simple LSTM) en 1.0x antes de evaluar.
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
        Console.WriteLine($"=== KAIROS103 BENCHMARK PROTOCOL ===");
        Console.WriteLine($"Dataset: {targetDataset.ToUpperInvariant()} | Directorio: {resolvedDir}");
        if (fastMode)
        {
            Console.WriteLine($"Modo: RÁPIDO (--fast) activado para verificación ágil.");
        }
        Console.WriteLine();

        string subDir = targetDataset.Equals("ecg", StringComparison.OrdinalIgnoreCase) ? "ecg" : "synthetic";
        string trainFile = Path.Combine(resolvedDir, subDir, subDir == "ecg" ? "ecg_train.csv" : "warpstress_train.csv");

        if (!File.Exists(trainFile))
        {
            Console.WriteLine($"[ERROR] No se encontró el dataset de entrenamiento en: {trainFile}");
            Console.WriteLine($"        Ruta de búsqueda actual: {Directory.GetCurrentDirectory()}");
            return ([], [], [], [], new EfficiencyTracker());
        }

        Console.WriteLine($"[1/4] Cargando dataset de entrenamiento nominal (1.0x)...");
        var trainData = DatasetLoader.LoadCsv(trainFile);
        if (fastMode && trainData.Count > 400)
        {
            trainData = trainData.Take(400).ToList();
        }
        int classCount = trainData.Select(d => d.Label).Distinct().Count();
        Console.WriteLine($"      Muestras cargadas: {trainData.Count} | Clases: {classCount}");

        Console.WriteLine($"\n[2/4] Entrenando Modelos en Velocidad Nominal (1.0x)...");
        
        Console.Write($"      [1/3] Entrenando Kairos103 (Rank-Order Modular N=4)... ");
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
                Console.Write($"\r      [1/3] Entrenando Kairos103 (Rank-Order Modular N=4)... Época {curr}/{tot}");
            }
        });
        Console.WriteLine(" -> Completado.");

        int tfEpochs = fastMode ? 8 : 12;
        Console.Write($"      [2/3] Entrenando Baseline Mini-Transformer (Attention + PE con Adam)... ");
        var transformer = new MiniTransformer(classCount);
        transformer.Train(trainData, epochs: tfEpochs, lr: 0.01, (curr, tot) =>
        {
            Console.Write($"\r      [2/3] Entrenando Baseline Mini-Transformer (Attention + PE con Adam)... Época {curr}/{tot}");
        });
        Console.WriteLine(" -> Completado.");

        int lstmEpochs = fastMode ? 8 : 12;
        Console.Write($"      [3/3] Entrenando Baseline Simple LSTM (BPTT con Adam)... ");
        var lstm = new SimpleLstm(classCount);
        lstm.Train(trainData, epochs: lstmEpochs, lr: 0.01, (curr, tot) =>
        {
            Console.Write($"\r      [3/3] Entrenando Baseline Simple LSTM (BPTT con Adam)... Época {curr}/{tot}");
        });
        Console.WriteLine(" -> Completado.");

        double[] factors = customFactors ?? [0.5, 0.8, 1.0, 1.2, 2.0];
        double[] kairosAcc = new double[factors.Length];
        double[] transformerAcc = new double[factors.Length];
        double[] lstmAcc = new double[factors.Length];

        var effTracker = new EfficiencyTracker();
        effTracker.StartMemoryTracking();

        Console.WriteLine($"\n[3/4] Evaluando estrés temporal bajo factores: {string.Join(", ", factors.Select(f => f.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture) + "x"))}");

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
                Console.WriteLine($"      [AVISO] Archivo de test no hallado para factor {fStr}x: {testFile}");
                continue;
            }

            var testData = DatasetLoader.LoadCsv(testFile);
            if (fastMode && testData.Count > 200)
            {
                testData = testData.Take(200).ToList();
            }

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

            kairosAcc[i] = (double)correctKairos / testData.Count;
            transformerAcc[i] = (double)correctTrans / testData.Count;
            lstmAcc[i] = (double)correctLstm / testData.Count;

            Console.WriteLine($"      Factor {fStr}x -> Kairos: {kairosAcc[i] * 100.0:F1}% | Transf: {transformerAcc[i] * 100.0:F1}% | LSTM: {lstmAcc[i] * 100.0:F1}%");
        }

        effTracker.StopMemoryTracking();

        Console.WriteLine($"\n[4/4] REPORTE DE RESULTADOS");
        string mdTable = AsciiPlotter.GenerateMarkdownTable(targetDataset, factors, kairosAcc, transformerAcc, lstmAcc);
        Console.WriteLine(mdTable);

        Console.WriteLine($"Métricas de Eficiencia Computacional:");
        Console.WriteLine($"  - Kairos Front-End MACs (continuos):  {effTracker.TotalFrontEndMacs:N0}");
        Console.WriteLine($"  - Kairos Spikes Discretos (eventos):   {effTracker.TotalSpikes:N0}");
        Console.WriteLine($"  - Kairos Acumulaciones (clasificador): {effTracker.TotalAccumulateOps:N0}");
        Console.WriteLine($"  - Mini-Transformer MACs (continuos):   {effTracker.TotalTransformerMacs:N0}");
        Console.WriteLine($"  - Simple LSTM MACs (continuos):        {effTracker.TotalLstmMacs:N0}");
        Console.WriteLine($"  - Memoria Heap Asignada en Thread:     {effTracker.MemoryAllocatedBytes:N0} bytes");

        return (factors, kairosAcc, transformerAcc, lstmAcc, effTracker);
    }

    private static string LocateDatasetsDirectory(string specifiedDir)
    {
        if (Directory.Exists(specifiedDir))
            return Path.GetFullPath(specifiedDir);

        // Buscar hacia arriba en el árbol de carpetas desde Environment.CurrentDirectory y AppContext.BaseDirectory
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
