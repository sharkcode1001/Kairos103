using System.Diagnostics;
using Kairos103.Core.Data;
using Kairos103.Core.Evaluation;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

public static class ValidateEcgCommand
{
    public static void Execute(
        string modelPath,
        string trainPath,
        double valSplit = 0.2,
        int seed = 42)
    {
        Console.WriteLine("\n==========================================================================");
        Console.WriteLine("          KAIROS103 - KAGGLE ECG VALIDATION EVALUATOR                     ");
        Console.WriteLine("==========================================================================");

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"[ERROR] Model file not found: {modelPath}");
            return;
        }

        if (!File.Exists(trainPath))
        {
            Console.WriteLine($"[ERROR] Data file not found: {trainPath}");
            return;
        }

        Console.WriteLine($"Loading model from: {modelPath}...");
        var classifier = KairosClassifier.Load(modelPath);
        Console.WriteLine($"  Model Architecture: Kairos103 | Subregions: {classifier.SubregionCount} | Classes: {classifier.ClassCount} | Readout Dim: {classifier.HiddenDim}");

        Console.WriteLine($"Loading dataset from: {trainPath}...");
        var sw = Stopwatch.StartNew();
        var allSamples = KaggleEcgDataset.LoadTrain(trainPath);
        sw.Stop();
        Console.WriteLine($"Loaded {allSamples.Count:N0} samples in {sw.ElapsedMilliseconds} ms.");

        IReadOnlyList<KaggleEcgSample> evalSamples = allSamples;
        if (valSplit > 0.0 && valSplit < 1.0)
        {
            Console.WriteLine($"Selecting validation split (Ratio: {valSplit:P0}, Seed: {seed})...");
            var (_, val) = KaggleEcgDataset.StratifiedSplit(allSamples, valSplit, seed);
            evalSamples = val;
            Console.WriteLine($"  Validation evaluation fold: {evalSamples.Count:N0} samples");
        }
        else
        {
            Console.WriteLine($"Evaluating on full dataset: {evalSamples.Count:N0} samples");
        }

        sw.Restart();
        var actuals = new int[evalSamples.Count];
        var preds = new int[evalSamples.Count];

        for (int i = 0; i < evalSamples.Count; i++)
        {
            var s = evalSamples[i];
            actuals[i] = s.Label;
            preds[i] = classifier.Predict(s.ToCombinedSignal(), effectiveLength: KaggleEcgSample.WaveformLength);
        }
        sw.Stop();

        Console.WriteLine($"Inference completed in {sw.ElapsedMilliseconds} ms ({(double)sw.ElapsedMilliseconds / evalSamples.Count * 1000.0:F2} µs/sample).\n");

        var metrics = MacroF1Metrics.Compute(actuals, preds, classifier.ClassCount);
        Console.WriteLine(metrics.FormatReport());
    }
}
