using System.Diagnostics;
using Kairos103.Cli.Localization;
using Kairos103.Core.Data;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

public static class TrainCommand
{
    public static void Execute(
        string dataPath,
        string outputPath,
        int subregions = 16,
        int epochs = 25,
        double lr = 0.05,
        int hiddenDim = 32,
        int seed = 20261010)
    {
        Console.WriteLine(CliStrings.IsSpanish
            ? $"\n=== ENTRENAMIENTO DE CAJA NEGRA KAIROS103 ==="
            : $"\n=== KAIROS103 BLACK-BOX TRAINING ===");

        if (!File.Exists(dataPath))
        {
            Console.WriteLine(CliStrings.IsSpanish
                ? $"[ERROR] Archivo de datos no encontrado: {dataPath}"
                : $"[ERROR] Data file not found: {dataPath}");
            return;
        }

        Console.WriteLine(CliStrings.IsSpanish
            ? $"Cargando dataset desde: {dataPath}..."
            : $"Loading dataset from: {dataPath}...");

        var sw = Stopwatch.StartNew();
        var samples = SignalDataset.LoadCsv(dataPath);
        sw.Stop();

        int classCount = samples.Select(s => s.Label).Distinct().Count();
        Console.WriteLine(CliStrings.IsSpanish
            ? $"  Muestras: {samples.Count} | Clases detectadas: {classCount} | Longitud media: {samples.Average(s => s.EffectiveLength):F0} pts (Carga: {sw.ElapsedMilliseconds} ms)"
            : $"  Samples: {samples.Count} | Detected Classes: {classCount} | Mean Length: {samples.Average(s => s.EffectiveLength):F0} pts (Loaded in: {sw.ElapsedMilliseconds} ms)");

        string readoutDesc = hiddenDim > 0
            ? (CliStrings.IsSpanish ? $"{hiddenDim} neuronas ocultas (Readout No-Lineal Adam)" : $"{hiddenDim} hidden units (Adam Non-Linear Readout)")
            : (CliStrings.IsSpanish ? "Readout Lineal" : "Linear Readout");

        Console.WriteLine(CliStrings.IsSpanish
            ? $"Configuración de red: {subregions} subregiones ({subregions * 4} neuronas LIF, {subregions * 24} rasgos Lehmer, {readoutDesc})"
            : $"Network config: {subregions} subregions ({subregions * 4} LIF neurons, {subregions * 24} Lehmer features, {readoutDesc})");

        var classifier = new KairosClassifier(subregionCount: subregions, seed: seed);

        Console.Write(CliStrings.IsSpanish ? "Entrenando modelo: " : "Training model: ");
        sw.Restart();
        classifier.Fit(samples, epochs: epochs, lr: lr, hiddenDim: hiddenDim, onProgress: (curr, tot) =>
        {
            if (curr == 1 || curr == tot || curr % 5 == 0)
            {
                Console.Write(CliStrings.IsSpanish ? $"\rEntrenando modelo: Época {curr}/{tot}" : $"\rTraining model: Epoch {curr}/{tot}");
            }
        });
        sw.Stop();
        Console.WriteLine(CliStrings.IsSpanish ? $" -> LISTO en {sw.ElapsedMilliseconds} ms." : $" -> DONE in {sw.ElapsedMilliseconds} ms.");

        // Self-evaluation on training set
        var eval = classifier.Evaluate(samples);
        Console.WriteLine(CliStrings.IsSpanish
            ? $"\nPrecisión en entrenamiento: {eval.Accuracy * 100:F2}% ({eval.CorrectSamples}/{eval.TotalSamples})"
            : $"\nTraining Set Accuracy: {eval.Accuracy * 100:F2}% ({eval.CorrectSamples}/{eval.TotalSamples})");
        Console.WriteLine(CliStrings.IsSpanish
            ? $"Eventos por inferencia: ~{eval.AvgSpikesPerSample:F1} spikes (sumas condicionales) | {eval.AvgFrontEndMacsPerSample:N0} MACs front-end"
            : $"Events per inference: ~{eval.AvgSpikesPerSample:F1} spikes (conditional adds) | {eval.AvgFrontEndMacsPerSample:N0} front-end MACs");

        // Save model
        string dir = Path.GetDirectoryName(outputPath) ?? "";
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        classifier.Save(outputPath);
        Console.WriteLine(CliStrings.IsSpanish
            ? $"Modelo guardado exitosamente en: {outputPath}\n"
            : $"Model successfully saved to: {outputPath}\n");
    }
}
