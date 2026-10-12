using System.Diagnostics;
using System.Text.Json;
using Kairos103.Cli.Localization;
using Kairos103.Core.Data;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

public static class EvaluateCommand
{
    public static void Execute(string modelPath, string dataPath, string? exportJsonPath = null)
    {
        Console.WriteLine(CliStrings.IsSpanish
            ? $"\n=== EVALUACIÓN DETALLADA KAIROS103 ==="
            : $"\n=== KAIROS103 DETAILED EVALUATION ===");

        if (!File.Exists(modelPath) || !File.Exists(dataPath))
        {
            Console.WriteLine(CliStrings.IsSpanish
                ? "[ERROR] Verifique las rutas del modelo y de los datos."
                : "[ERROR] Verify paths for both model and data files.");
            return;
        }

        var classifier = KairosClassifier.Load(modelPath);
        var samples = SignalDataset.LoadCsv(dataPath);

        var sw = Stopwatch.StartNew();
        var eval = classifier.Evaluate(samples);
        sw.Stop();

        double usPerInference = (sw.Elapsed.TotalMilliseconds * 1000.0) / eval.TotalSamples;

        Console.WriteLine(CliStrings.IsSpanish
            ? $"\nExactitud Global (Accuracy): {eval.Accuracy * 100:F2}% ({eval.CorrectSamples}/{eval.TotalSamples})"
            : $"\nGlobal Accuracy: {eval.Accuracy * 100:F2}% ({eval.CorrectSamples}/{eval.TotalSamples})");
        Console.WriteLine(CliStrings.IsSpanish
            ? $"Latencia media: {usPerInference:F1} µs | Spikes medios: ~{eval.AvgSpikesPerSample:F1} eventos | MACs front-end: {eval.AvgFrontEndMacsPerSample:N0}"
            : $"Mean Latency: {usPerInference:F1} µs | Mean Spikes: ~{eval.AvgSpikesPerSample:F1} events | Front-End MACs: {eval.AvgFrontEndMacsPerSample:N0}");

        // Print Confusion Matrix
        Console.WriteLine(CliStrings.IsSpanish ? "\nMatriz de Confusión (Filas = Real, Columnas = Predicho):" : "\nConfusion Matrix (Rows = True, Columns = Pred):");
        int k = classifier.ClassCount;

        Console.Write("       ");
        for (int c = 0; c < k; c++) Console.Write($"  C{c,-3}");
        Console.WriteLine();

        for (int r = 0; r < k; r++)
        {
            Console.Write($"  C{r,-3}:");
            for (int c = 0; c < k; c++)
            {
                Console.Write($"  {eval.ConfusionMatrix[r, c],-4}");
            }
            Console.WriteLine();
        }

        if (!string.IsNullOrEmpty(exportJsonPath))
        {
            var exportObj = new
            {
                Benchmark = "Kairos103-Evaluation",
                Date = DateTime.UtcNow.ToString("o"),
                Accuracy = eval.Accuracy,
                TotalSamples = eval.TotalSamples,
                CorrectSamples = eval.CorrectSamples,
                AvgSpikesPerDecision = eval.AvgSpikesPerSample,
                AvgFrontEndMacs = eval.AvgFrontEndMacsPerSample,
                LatencyMicroseconds = usPerInference,
                ConfusionMatrix = eval.ConfusionMatrix
            };

            string json = JsonSerializer.Serialize(exportObj, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(exportJsonPath, json);
            Console.WriteLine(CliStrings.IsSpanish
                ? $"\nMétricas exportadas en formato JSON para benchmark: {exportJsonPath}\n"
                : $"\nMetrics exported to JSON benchmark format: {exportJsonPath}\n");
        }
    }
}
