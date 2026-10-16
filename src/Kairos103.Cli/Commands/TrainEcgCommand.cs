using System.Diagnostics;
using System.Globalization;
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
        bool useOvr = false,
        double[]? weightPowers = null,
        int cvFolds = 0)
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

        // K-Fold Cross Validation mode
        if (cvFolds >= 2)
        {
            ExecuteKFoldsOof(allSamples, modelPath, cvFolds, epochs, lr, hiddenDim, subregions, seed, useOvr, weightPower, weightPowers);
            return;
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
            if (weightPowers != null && weightPowers.Length == 4)
            {
                Console.WriteLine($"\nAsymmetric Class Loss Weights (Powers: [{string.Join(", ", weightPowers.Select(p => p.ToString("F2", CultureInfo.InvariantCulture)))}]):");
                for (int c = 0; c < 4; c++)
                {
                    var w = KaggleEcgDataset.ComputeBalancedClassWeights(trainSamples, 4, weightPowers[c]);
                    Console.WriteLine($"  Class {c}: {w[c]:F3}");
                }
            }
            else
            {
                classWeights = KaggleEcgDataset.ComputeBalancedClassWeights(trainSamples, 4, weightPower);
                Console.WriteLine($"\nBalanced Class Loss Weights (Power: {weightPower:F2}):");
                for (int c = 0; c < 4; c++)
                {
                    Console.WriteLine($"  Class {c}: {classWeights[c]:F3}");
                }
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
        Console.WriteLine($"  Auxiliary:  18 relative clinical features (3 rhythm + 6 morphology + 9 electrophysiology & ST-T discordance)");
        Console.WriteLine($"  Front-End:  4 Anatomical Banks (Global, P-Wave/PR, Central QRS, ST-Segment/T-Wave)");
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
                    defaultThresholds: [1.00, 1.00, 1.25, 1.00],
                    onProgress: (ep, totalEp) =>
                    {
                        Console.WriteLine($"  Epoch [{ep,2}/{totalEp,2}] complete on 100% training data (OvR).");
                    },
                    weightPowers: weightPowers);
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
                    calibrateThresholds: calibrateThresholds,
                    weightPowers: weightPowers);
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

        // Run ECG Diagnostics (CR-D1)
        EcgDiagnostics.ReportRPeakPlacement(allSamples);
        EcgDiagnostics.ReportTemplateSeparability(allSamples);

        Console.WriteLine("==========================================================================\n");
    }

    private static void ExecuteKFoldsOof(
        IReadOnlyList<KaggleEcgSample> allSamples,
        string modelPath,
        int cvFolds,
        int epochs,
        double lr,
        int hiddenDim,
        int subregions,
        int seed,
        bool useOvr,
        double weightPower,
        double[]? weightPowers)
    {
        Console.WriteLine($"\n[K-FOLD CV] Running {cvFolds}-Fold Stratified Cross-Validation (Seed: {seed})...");
        var folds = KaggleEcgDataset.StratifiedKFolds(allSamples, cvFolds, seed);
        var oofProbs = new double[allSamples.Count][];
        var oofActuals = new int[allSamples.Count];
        var foldModels = new List<string>();

        int oofSampleIdx = 0;
        double sumValMacroF1 = 0.0;

        for (int fold = 0; fold < cvFolds; fold++)
        {
            Console.WriteLine($"\n--------------------------------------------------------------------------");
            Console.WriteLine($"  FOLD {fold + 1}/{cvFolds} (Val Size: {folds[fold].Count:N0} samples)");
            Console.WriteLine($"--------------------------------------------------------------------------");

            var valSamplesFold = folds[fold];
            var trainSamplesFold = folds.Where((_, idx) => idx != fold).SelectMany(f => f).ToList();

            var trainSignals = trainSamplesFold.Select(s => s.ToSignalSample()).ToList();
            var valSignals = valSamplesFold.Select(s => s.ToSignalSample()).ToList();

            var foldClassifier = new KairosClassifier(subregionCount: subregions, seed: seed + fold);

            string dir = Path.GetDirectoryName(modelPath) ?? "";
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            string foldModelPath = modelPath.EndsWith(".kairos")
                ? modelPath.Substring(0, modelPath.Length - 7) + $"_fold{fold + 1}.kairos"
                : $"{modelPath}_fold{fold + 1}.kairos";

            MacroF1Result foldRes;
            if (useOvr)
            {
                foldRes = foldClassifier.FitOvrWithValidation(
                    trainSignals,
                    valSignals,
                    epochs: epochs,
                    lr: lr,
                    hiddenDim: hiddenDim,
                    weightPower: weightPower,
                    onEpochCompleted: (ep, totalEp, valAcc, valMacroF1) =>
                    {
                        if (ep == totalEp || ep % 5 == 0)
                            Console.WriteLine($"    Epoch [{ep,2}/{totalEp,2}] -> Val Macro F1: {valMacroF1:F4} | Val Acc: {valAcc * 100:F2}%");
                    },
                    calibrateThresholds: false,
                    weightPowers: weightPowers);
            }
            else
            {
                var weights = KaggleEcgDataset.ComputeBalancedClassWeights(trainSamplesFold, 4, weightPower);
                foldRes = foldClassifier.FitWithValidation(
                    trainSignals,
                    valSignals,
                    epochs: epochs,
                    lr: lr,
                    hiddenDim: hiddenDim,
                    classWeights: weights,
                    onEpochCompleted: (ep, totalEp, valAcc, valMacroF1) =>
                    {
                        if (ep == totalEp || ep % 5 == 0)
                            Console.WriteLine($"    Epoch [{ep,2}/{totalEp,2}] -> Val Macro F1: {valMacroF1:F4} | Val Acc: {valAcc * 100:F2}%");
                    },
                    calibrateThresholds: false);
            }

            sumValMacroF1 += foldRes.MacroF1;
            foldClassifier.Save(foldModelPath);
            foldModels.Add(foldModelPath);
            Console.WriteLine($"  Fold {fold + 1} Best Macro F1: {foldRes.MacroF1:F4} (Saved: {foldModelPath})");

            for (int i = 0; i < valSignals.Count; i++)
            {
                oofProbs[oofSampleIdx] = foldClassifier.PredictProbabilities(valSignals[i].Values, valSignals[i].EffectiveLength);
                oofActuals[oofSampleIdx] = valSignals[i].Label;
                oofSampleIdx++;
            }
        }

        double meanFoldF1 = sumValMacroF1 / cvFolds;
        Console.WriteLine($"\n==========================================================================");
        Console.WriteLine($"  {cvFolds}-FOLD CV COMPLETE. Mean Fold Macro F1: {meanFoldF1:F4}");
        Console.WriteLine($"==========================================================================");

        Console.WriteLine("\nCalibrating Global Decision Thresholds on Out-Of-Fold (OOF) Probabilities...");
        var candidates = new double[] { 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 1.0, 1.10, 1.25, 1.50, 1.80, 2.20, 2.60, 3.0 };
        double[] oofThresholds = Enumerable.Repeat(1.0, 4).ToArray();
        var oofRes = EvaluateProbsWithThresholds(oofProbs, oofActuals, oofThresholds);
        double bestOofF1 = oofRes.MacroF1;
        double[] bestThresh = (double[])oofThresholds.Clone();

        for (int iter = 0; iter < 4; iter++)
        {
            for (int c = 0; c < 4; c++)
            {
                double bestCand = bestThresh[c];
                foreach (double cand in candidates)
                {
                    bestThresh[c] = cand;
                    var res = EvaluateProbsWithThresholds(oofProbs, oofActuals, bestThresh);
                    if (res.MacroF1 > bestOofF1)
                    {
                        bestOofF1 = res.MacroF1;
                        bestCand = cand;
                    }
                }
                bestThresh[c] = bestCand;
            }
        }

        var finalOofResult = EvaluateProbsWithThresholds(oofProbs, oofActuals, bestThresh);
        Console.WriteLine($"\nOptimal Out-Of-Fold Thresholds:");
        Console.WriteLine($"  Thresholds: [{string.Join(", ", bestThresh.Select((t, c) => $"Class {c}={t:F2}"))}]");
        Console.WriteLine($"  Raw OOF Macro F1 (at 1.0): {oofRes.MacroF1:F4}");
        Console.WriteLine($"  Calibrated Global OOF Macro F1: {finalOofResult.MacroF1:F4} (Accuracy: {finalOofResult.Accuracy * 100:F2}%)");
        Console.WriteLine($"\nOOF Full Classification Report:");
        Console.WriteLine(finalOofResult.FormatReport());

        foreach (var fPath in foldModels)
        {
            var mod = KairosClassifier.Load(fPath);
            mod.Thresholds = bestThresh;
            mod.Save(fPath);
        }

        Console.WriteLine($"All {cvFolds} fold models updated with optimal OOF thresholds.");
        Console.WriteLine($"\nTo generate ensemble submission on test.csv:");
        Console.WriteLine($"  kairos predict-kaggle --model {string.Join(",", foldModels)} --test test.csv --out submission.csv");

        // Run ECG Diagnostics (CR-D1)
        EcgDiagnostics.ReportRPeakPlacement(allSamples);
        EcgDiagnostics.ReportTemplateSeparability(allSamples);

        Console.WriteLine("==========================================================================\n");
    }

    private static MacroF1Result EvaluateProbsWithThresholds(double[][] probs, int[] actuals, double[] thresh)
    {
        int[] preds = new int[actuals.Length];
        for (int i = 0; i < actuals.Length; i++)
        {
            int best = 0;
            double bestScore = probs[i][0] / Math.Max(1e-4, thresh[0]);
            for (int c = 1; c < 4; c++)
            {
                double s = probs[i][c] / Math.Max(1e-4, thresh[c]);
                if (s > bestScore)
                {
                    bestScore = s;
                    best = c;
                }
            }
            preds[i] = best;
        }
        return MacroF1Metrics.Compute(actuals, preds, 4);
    }
}
