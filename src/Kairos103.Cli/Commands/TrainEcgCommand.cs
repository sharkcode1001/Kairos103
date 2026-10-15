using System.Diagnostics;
using Kairos103.Core.Baselines;
using Kairos103.Core.Data;
using Kairos103.Core.Evaluation;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

public static class TrainEcgCommand
{
    public static void Execute(
        string trainPath,
        string modelPath,
        int epochs = 35,
        double lr = 0.02,
        int hiddenDim = 64,
        int subregions = 32,
        double valSplit = 0.2,
        int seed = 42,
        bool useClassWeights = true,
        bool runBaselines = true,
        double weightPower = 0.5,
        bool calibrateThresholds = true,
        bool useOvr = false)
    {
        Console.WriteLine("\n==========================================================================");
        Console.WriteLine("          KAIROS103 - KAGGLE ECG ARRHYTHMIA TRAINING PIPELINE             ");
        Console.WriteLine("==========================================================================");

        if (!File.Exists(trainPath))
        {
            Console.WriteLine($"[ERROR] Training file not found: {trainPath}");
            return;
        }

        var sw = Stopwatch.StartNew();
        Console.WriteLine($"Loading Kaggle train dataset from: {trainPath}...");
        var allSamples = KaggleEcgDataset.LoadTrain(trainPath);
        sw.Stop();

        Console.WriteLine($"Loaded {allSamples.Count:N0} samples in {sw.ElapsedMilliseconds} ms.");

        // Report class distribution
        var classCounts = new int[4];
        foreach (var s in allSamples)
        {
            if (s.Label >= 0 && s.Label < 4) classCounts[s.Label]++;
        }

        Console.WriteLine("\nDataset Class Distribution:");
        string[] classNames = ["0 (Normal)", "1 (SVEB)", "2 (VEB)", "3 (Fusion)"];
        for (int c = 0; c < 4; c++)
        {
            double pct = (double)classCounts[c] / allSamples.Count * 100.0;
            Console.WriteLine($"  Class {classNames[c],-12}: {classCounts[c],7:N0} samples ({pct,5:F2}%)");
        }

        // Stratified split
        Console.WriteLine($"\nPerforming stratified train/val split (Val Ratio: {valSplit:P0}, Seed: {seed})...");
        var (trainSamples, valSamples) = KaggleEcgDataset.StratifiedSplit(allSamples, valSplit, seed);
        Console.WriteLine($"  Train split: {trainSamples.Count:N0} samples");
        Console.WriteLine($"  Val split:   {valSamples.Count:N0} samples");

        // Optional baseline evaluation
        if (runBaselines)
        {
            Console.WriteLine("\nEvaluating Reference Baselines on Validation Set:");
            Console.WriteLine("--------------------------------------------------------------------------");
            IEcgBaseline[] baselines = [
                new MajorityDummyClassifier(),
                new LinearEcgClassifier(),
                new Cnn1DEcgClassifier(seed)
            ];

            var valActuals = valSamples.Select(s => s.Label).ToArray();

            foreach (var baseline in baselines)
            {
                var bSw = Stopwatch.StartNew();
                baseline.Fit(trainSamples);
                bSw.Stop();

                var preds = valSamples.Select(baseline.Predict).ToArray();
                var bMetrics = MacroF1Metrics.Compute(valActuals, preds, 4);

                Console.WriteLine($"  [{baseline.Name}]");
                Console.WriteLine($"    Macro F1: {bMetrics.MacroF1:F4} | Accuracy: {bMetrics.Accuracy * 100:F2}% | F1: [0]={bMetrics.ClassF1[0]:F3}, [1]={bMetrics.ClassF1[1]:F3}, [2]={bMetrics.ClassF1[2]:F3}, [3]={bMetrics.ClassF1[3]:F3} ({bSw.ElapsedMilliseconds} ms)");
            }
            Console.WriteLine("--------------------------------------------------------------------------");
        }

        // Compute balanced class weights if requested
        double[]? classWeights = null;
        if (useClassWeights)
        {
            classWeights = KaggleEcgDataset.ComputeBalancedClassWeights(trainSamples, 4, weightPower);
            Console.WriteLine($"\nBalanced Class Loss Weights (Power: {weightPower:F2}):");
            for (int c = 0; c < 4; c++)
            {
                Console.WriteLine($"  Class {c}: {classWeights[c]:F3}");
            }
        }

        // Convert to SignalSample with EffectiveLength = 250 (waveform length)
        var trainSignalSamples = trainSamples.Select(s => s.ToSignalSample()).ToList();
        var valSignalSamples = valSamples.Select(s => s.ToSignalSample()).ToList();

        Console.WriteLine($"\nInitializing Kairos103 Classifier:");
        Console.WriteLine($"  Subregions: {subregions} ({subregions * 4} LIF neurons, {subregions * 24} Lehmer features)");
        Console.WriteLine(useOvr 
            ? $"  Readout:    One-vs-Rest Binary Heads ({hiddenDim} hidden neurons/head, Adam optimizer)" 
            : $"  Readout:    2-layer MLP ({hiddenDim} hidden neurons, Adam optimizer)");
        Console.WriteLine($"  Waveform:   Per-beat min-max normalized [0, 1] (patient-independent scale)");
        Console.WriteLine($"  Auxiliary:  9 relative features (3 rhythm + 6 dimensionless morphological & pause ratios)");
        Console.WriteLine($"  Optimization Metric: Macro F1 on validation fold\n");

        var classifier = new KairosClassifier(subregionCount: subregions, seed: seed);

        if (useOvr)
        {
            if (valSamples.Count == 0)
            {
                sw.Restart();
                classifier.FitOvrFull(
                    trainSignalSamples,
                    epochs: epochs > 0 ? epochs : 28,
                    lr: lr,
                    hiddenDim: hiddenDim,
                    weightPower: weightPower,
                    defaultThresholds: [1.0, 1.0, 1.0, 1.0],
                    onProgress: (ep, totalEp) =>
                    {
                        Console.WriteLine($"  Epoch [{ep,2}/{totalEp,2}] complete on 100% training data (OvR).");
                    });
                sw.Stop();

                Console.WriteLine($"\nOvR Full Training completed in {sw.ElapsedMilliseconds} ms.");
                Console.WriteLine($"Applying OvR decision thresholds: [{string.Join(", ", classifier.Thresholds.Select((t, c) => $"Class {c}={t:F2}"))}]\n");
            }
            else
            {
                sw.Restart();
                var bestResult = classifier.FitOvrWithValidation(
                    trainSignalSamples,
                    valSignalSamples,
                    epochs: epochs,
                    lr: lr,
                    hiddenDim: hiddenDim,
                    weightPower: weightPower,
                    onEpochCompleted: (ep, totalEp, valAcc, valMacroF1) =>
                    {
                        Console.WriteLine($"  Epoch [{ep,2}/{totalEp,2}] -> Val Macro F1: {valMacroF1:F4} | Val Accuracy: {valAcc * 100:F2}%");
                    },
                    calibrateThresholds: calibrateThresholds);
                sw.Stop();

                Console.WriteLine($"\nOvR Training completed in {sw.ElapsedMilliseconds} ms.");
                Console.WriteLine($"Best Validation Macro F1 Checkpoint Restored: {bestResult.MacroF1:F4}\n");

                // Print final classification report
                Console.WriteLine(bestResult.FormatReport());
            }
        }
        else
        {
            if (valSamples.Count == 0)
            {
                sw.Restart();
                classifier.FitFull(
                    trainSignalSamples,
                    epochs: epochs > 0 ? epochs : 28,
                    lr: lr,
                    hiddenDim: hiddenDim,
                    classWeights: classWeights,
                    defaultThresholds: [0.50, 0.18, 0.90, 1.00],
                    onProgress: (ep, totalEp) =>
                    {
                        Console.WriteLine($"  Epoch [{ep,2}/{totalEp,2}] complete on 100% training data.");
                    });
                sw.Stop();

                Console.WriteLine($"\nFull Training completed in {sw.ElapsedMilliseconds} ms.");
                Console.WriteLine($"Applying proven optimal thresholds: [{string.Join(", ", classifier.Thresholds.Select((t, c) => $"Class {c}={t:F2}"))}]\n");
            }
            else
            {
                sw.Restart();
                var bestResult = classifier.FitWithValidation(
                    trainSignalSamples,
                    valSignalSamples,
                    epochs: epochs,
                    lr: lr,
                    hiddenDim: hiddenDim,
                    classWeights: classWeights,
                    onEpochCompleted: (ep, totalEp, valAcc, valMacroF1) =>
                    {
                        Console.WriteLine($"  Epoch [{ep,2}/{totalEp,2}] -> Val Macro F1: {valMacroF1:F4} | Val Accuracy: {valAcc * 100:F2}%");
                    },
                    calibrateThresholds: calibrateThresholds);
                sw.Stop();

                Console.WriteLine($"\nTraining completed in {sw.ElapsedMilliseconds} ms.");
                Console.WriteLine($"Best Validation Macro F1 Checkpoint Restored: {bestResult.MacroF1:F4}\n");

                // Print final classification report
                Console.WriteLine(bestResult.FormatReport());
            }
        }

        // Save model
        string dir = Path.GetDirectoryName(modelPath) ?? "";
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        classifier.Save(modelPath);
        Console.WriteLine($"Model checkpoint successfully saved to: {modelPath}");
        Console.WriteLine("==========================================================================\n");
    }
}
