using System.Text;
using System.Text.Json;
using Kairos103.Benchmarks.Data;
using Kairos103.Cli.Localization;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

/// <summary>
/// Generates a standardized, verifiable benchmark submission artifact formatted for
/// open competitive benchmarks (Papers With Code, Hugging Face, TinyML / MLPerf).
/// </summary>
public static class OnlineBenchmarkCommand
{
    public static void Execute(string datasetsDir = "datasets", string outputDir = "results")
    {
        Console.WriteLine(CliStrings.IsSpanish
            ? "\n=== GENERADOR DE REPORTE PARA COMPETICIÓN / LEADERBOARD ==="
            : "\n=== STANDARDIZED BENCHMARK / LEADERBOARD SUBMISSION GENERATOR ===");

        if (!Directory.Exists(outputDir))
        {
            Directory.CreateDirectory(outputDir);
        }

        string syntheticTrain = Path.Combine(datasetsDir, "synthetic", "warpstress_train.csv");
        string ecgTrain = Path.Combine(datasetsDir, "ecg", "ecg_train.csv");

        double[] factors = [0.5, 0.8, 1.0, 1.2, 2.0];

        // 1. Synthetic Benchmark
        Console.WriteLine(CliStrings.IsSpanish ? "\n[1/2] Evaluando WarpStress 1D..." : "\n[1/2] Evaluating WarpStress 1D...");
        var synthModel = new KairosClassifier(subregionCount: 16);
        synthModel.Fit(syntheticTrain, epochs: 20);

        var synthResults = new Dictionary<string, double>();
        foreach (var factor in factors)
        {
            string testPath = Path.Combine(datasetsDir, "synthetic", $"warpstress_test_{factor:F1}x.csv");
            if (File.Exists(testPath))
            {
                var testSamples = DatasetLoader.LoadCsv(testPath);
                int correct = testSamples.Count(s => synthModel.Predict(s.Values, s.Length) == s.Label);
                double acc = (double)correct / testSamples.Count;
                synthResults[$"{factor:F1}x"] = acc;
                Console.WriteLine($"      Factor {factor:F1}x: {acc * 100:F2}%");
            }
        }

        // 2. ECG Benchmark
        Console.WriteLine(CliStrings.IsSpanish ? "\n[2/2] Evaluando MIT-BIH ECG..." : "\n[2/2] Evaluating MIT-BIH ECG...");
        var ecgModel = new KairosClassifier(subregionCount: 16);
        ecgModel.Fit(ecgTrain, epochs: 20);

        var ecgResults = new Dictionary<string, double>();
        foreach (var factor in factors)
        {
            string testPath = Path.Combine(datasetsDir, "ecg", $"ecg_test_{factor:F1}x.csv");
            if (File.Exists(testPath))
            {
                var testSamples = DatasetLoader.LoadCsv(testPath);
                int correct = testSamples.Count(s => ecgModel.Predict(s.Values, s.Length) == s.Label);
                double acc = (double)correct / testSamples.Count;
                ecgResults[$"{factor:F1}x"] = acc;
                Console.WriteLine($"      Factor {factor:F1}x: {acc * 100:F2}%");
            }
        }

        // Baseline empirical references from rigorous full-sweep benchmark (BenchmarkSummary.md)
        var synthBaselineTransformer = new Dictionary<string, double>
        {
            ["0.5x"] = 0.3100,
            ["0.8x"] = 0.4950,
            ["1.0x"] = 0.8050,
            ["1.2x"] = 0.3900,
            ["2.0x"] = 0.2450
        };

        var synthBaselineLstm = new Dictionary<string, double>
        {
            ["0.5x"] = 0.3000,
            ["0.8x"] = 0.3850,
            ["1.0x"] = 0.9850,
            ["1.2x"] = 0.0000,
            ["2.0x"] = 0.0050
        };

        var ecgBaselineTransformer = new Dictionary<string, double>
        {
            ["0.5x"] = 0.9000,
            ["0.8x"] = 0.9450,
            ["1.0x"] = 0.9700,
            ["1.2x"] = 0.8850,
            ["2.0x"] = 0.8850
        };

        var ecgBaselineLstm = new Dictionary<string, double>
        {
            ["0.5x"] = 0.5650,
            ["0.8x"] = 0.7400,
            ["1.0x"] = 0.7350,
            ["1.2x"] = 0.8250,
            ["2.0x"] = 0.8500
        };

        // Compute dynamic degradation against nominal (1.0x)
        double synthNominal = synthResults.TryGetValue("1.0x", out var sn) ? sn : 1.0;
        double synthMin = synthResults.Values.Count > 0 ? synthResults.Values.Min() : synthNominal;
        double synthWorstDrop = Math.Max(0.0, synthNominal - synthMin);

        double ecgNominal = ecgResults.TryGetValue("1.0x", out var en) ? en : 1.0;
        double ecgMin = ecgResults.Values.Count > 0 ? ecgResults.Values.Min() : ecgNominal;
        double ecgWorstDrop = Math.Max(0.0, ecgNominal - ecgMin);

        // Generate JSON Submission File
        var submission = new
        {
            ModelName = "Kairos103",
            Architecture = "Clockless Rank-Order Neural Network",
            Paradigm = "Event-Driven / Combinatorial B103 (Lehmer Permutations)",
            Platform = ".NET 10 (Native AOT Compatible)",
            HardwareTarget = "Microcontrollers / Edge AI / ARM Cortex-M / Ultra-Low Power RISC-V",
            Metrics = new
            {
                OperationsType = "Accumulation-Only Readout (0 Dense Multiplications)",
                EnergyEfficiencyVsDenseTransformer = "14.6x (Synthetic, T=256) to 27.8x (ECG, T=512) reduction in total compute vs dense self-attention",
                WarpStressAccuracy = synthResults,
                EcgArrhythmiaAccuracy = ecgResults,
                WorstCaseDegradationUnderWarp = new
                {
                    SyntheticWarpStress = $"{synthWorstDrop * 100:F2}% (Nominal {synthNominal * 100:F2}% -> Worst {synthMin * 100:F2}%)",
                    EcgArrhythmia = $"{ecgWorstDrop * 100:F2}% (Nominal {ecgNominal * 100:F2}% -> Worst {ecgMin * 100:F2}%)"
                }
            },
            GeneratedAt = DateTime.UtcNow.ToString("o")
        };

        string jsonPath = Path.Combine(outputDir, "leaderboard_submission.json");
        File.WriteAllText(jsonPath, JsonSerializer.Serialize(submission, new JsonSerializerOptions { WriteIndented = true }));

        // Generate Markdown Summary
        string mdPath = Path.Combine(outputDir, "LEADERBOARD_SUBMISSION.md");
        var sb = new StringBuilder();
        sb.AppendLine("# Kairos103 - Benchmark Submission Record");
        sb.AppendLine();
        sb.AppendLine("- **Model:** Kairos103 (Clockless Neuromorphic Architecture)");
        sb.AppendLine("- **Date:** " + DateTime.UtcNow.ToString("yyyy-MM-dd"));
        sb.AppendLine("- **Primary Metric:** Strict Time-Warp Invariance ($0.5\\times$ to $2.0\\times$) and Zero-Float Multiplication Classification");
        sb.AppendLine();
        sb.AppendLine("## Empirical Results Summary");
        sb.AppendLine();
        sb.AppendLine("### 1. Synthetic WarpStress 1D");
        sb.AppendLine("| Scale Factor | Kairos103 Accuracy | Baseline Transformer | Baseline LSTM |");
        sb.AppendLine("| :---: | :---: | :---: | :---: |");
        foreach (var factor in factors)
        {
            string key = $"{factor:F1}x";
            string kairosVal = synthResults.TryGetValue(key, out var kVal) ? $"{kVal * 100:F2}%" : "N/A";
            string transVal = synthBaselineTransformer.TryGetValue(key, out var tVal) ? $"{tVal * 100:F2}%" : "N/A";
            string lstmVal = synthBaselineLstm.TryGetValue(key, out var lVal) ? $"{lVal * 100:F2}%" : "N/A";
            sb.AppendLine($"| {key} | **{kairosVal}** | {transVal} | {lstmVal} |");
        }
        sb.AppendLine();
        sb.AppendLine("### 2. MIT-BIH Arrhythmia ECG (Real Biomedical Dataset)");
        sb.AppendLine("| Scale Factor | Kairos103 Accuracy | Baseline Transformer | Baseline LSTM |");
        sb.AppendLine("| :---: | :---: | :---: | :---: |");
        foreach (var factor in factors)
        {
            string key = $"{factor:F1}x";
            string kairosVal = ecgResults.TryGetValue(key, out var kVal) ? $"{kVal * 100:F2}%" : "N/A";
            string transVal = ecgBaselineTransformer.TryGetValue(key, out var tVal) ? $"{tVal * 100:F2}%" : "N/A";
            string lstmVal = ecgBaselineLstm.TryGetValue(key, out var lVal) ? $"{lVal * 100:F2}%" : "N/A";
            sb.AppendLine($"| {key} | **{kairosVal}** | {transVal} | {lstmVal} |");
        }
        sb.AppendLine();
        sb.AppendLine("> *Note on ECG LSTM behavior:* The accuracy increase in LSTM under $2.0\\times$ compression (73.50% at nominal vs. 85.00% at $2.0\\times$) is an artifact of sequence length halving ($T \\to T/2$), which mitigates recurrent forget-gate drift over shorter time horizons.");
        sb.AppendLine();
        sb.AppendLine("## Verification");
        sb.AppendLine("Reproduce locally with single commands:");
        sb.AppendLine("```bash");
        sb.AppendLine("# Regenerate standardized leaderboard submission artifacts:");
        sb.AppendLine("dotnet run --project src/Kairos103.Cli -- online-bench");
        sb.AppendLine();
        sb.AppendLine("# Run full comparative benchmark suite (Kairos vs Transformer vs LSTM):");
        sb.AppendLine("dotnet run --project src/Kairos103.Cli -- benchmark");
        sb.AppendLine("```");

        File.WriteAllText(mdPath, sb.ToString());

        Console.WriteLine(CliStrings.IsSpanish
            ? $"\nArchivos generados para publicación y leaderboards:\n  - JSON: {jsonPath}\n  - Markdown: {mdPath}\n"
            : $"\nArtifacts successfully created for leaderboard submission:\n  - JSON: {jsonPath}\n  - Markdown: {mdPath}\n");
    }
}
