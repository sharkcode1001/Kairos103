using System.Diagnostics;
using System.Text;
using Kairos103.Cli.Localization;
using Kairos103.Core.Data;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

public static class PredictCommand
{
    public static void Execute(string modelPath, string dataPath, string? outputPath = null)
    {
        Console.WriteLine(CliStrings.IsSpanish
            ? $"\n=== INFERENCIA KAIROS103 (PREDICCIÓN) ==="
            : $"\n=== KAIROS103 INFERENCE (PREDICTION) ===");

        if (!File.Exists(modelPath))
        {
            Console.WriteLine(CliStrings.IsSpanish
                ? $"[ERROR] Archivo de modelo no encontrado: {modelPath}"
                : $"[ERROR] Model file not found: {modelPath}");
            return;
        }

        if (!File.Exists(dataPath))
        {
            Console.WriteLine(CliStrings.IsSpanish
                ? $"[ERROR] Archivo de datos no encontrado: {dataPath}"
                : $"[ERROR] Data file not found: {dataPath}");
            return;
        }

        Console.WriteLine(CliStrings.IsSpanish
            ? $"Cargando modelo desde: {modelPath}..."
            : $"Loading model from: {modelPath}...");

        var classifier = KairosClassifier.Load(modelPath);
        Console.WriteLine(CliStrings.IsSpanish
            ? $"  Topología: {classifier.SubregionCount} subregiones ({classifier.TotalNeurons} neuronas LIF) | Clases: {classifier.ClassCount}"
            : $"  Topology: {classifier.SubregionCount} subregions ({classifier.TotalNeurons} LIF neurons) | Classes: {classifier.ClassCount}");

        Console.WriteLine(CliStrings.IsSpanish
            ? $"Cargando muestras de prueba desde: {dataPath}..."
            : $"Loading test samples from: {dataPath}...");

        var samples = SignalDataset.LoadCsv(dataPath);
        Console.WriteLine(CliStrings.IsSpanish
            ? $"  {samples.Count} muestras cargadas para inferencia."
            : $"  {samples.Count} samples loaded for inference.");

        var sw = Stopwatch.StartNew();
        var predictions = new List<(int Index, int TrueLabel, int PredictedLabel, int Spikes, long Macs)>();

        for (int i = 0; i < samples.Count; i++)
        {
            var s = samples[i];
            var (pred, spikes, macs, _, _) = classifier.PredictDetailed(s.Values, s.EffectiveLength);
            predictions.Add((i, s.Label, pred, spikes, macs));
        }
        sw.Stop();

        double elapsedMs = sw.Elapsed.TotalMilliseconds;
        double usPerInference = (elapsedMs * 1000.0) / samples.Count;

        int correct = predictions.Count(p => p.TrueLabel == p.PredictedLabel);
        double accuracy = (double)correct / samples.Count;

        Console.WriteLine(CliStrings.IsSpanish
            ? $"\n--- Resultados de Inferencia ---"
            : $"\n--- Inference Results ---");
        Console.WriteLine(CliStrings.IsSpanish
            ? $"  Precisión:            {accuracy * 100:F2}% ({correct}/{samples.Count})"
            : $"  Accuracy:             {accuracy * 100:F2}% ({correct}/{samples.Count})");
        Console.WriteLine(CliStrings.IsSpanish
            ? $"  Tiempo total:         {elapsedMs:F2} ms ({usPerInference:F1} µs por inferencia)"
            : $"  Total Time:           {elapsedMs:F2} ms ({usPerInference:F1} µs per inference)");
        Console.WriteLine(CliStrings.IsSpanish
            ? $"  Spikes promedio:      {predictions.Average(p => p.Spikes):F1} eventos por decisión"
            : $"  Average Spikes:       {predictions.Average(p => p.Spikes):F1} events per decision");

        if (!string.IsNullOrEmpty(outputPath))
        {
            var sb = new StringBuilder();
            sb.AppendLine("Index,TrueLabel,PredictedLabel,Spikes,FrontEndMacs");
            foreach (var p in predictions)
            {
                sb.AppendLine($"{p.Index},{p.TrueLabel},{p.PredictedLabel},{p.Spikes},{p.Macs}");
            }
            File.WriteAllText(outputPath, sb.ToString());
            Console.WriteLine(CliStrings.IsSpanish
                ? $"Predicciones exportadas a: {outputPath}\n"
                : $"Predictions exported to: {outputPath}\n");
        }
    }
}
