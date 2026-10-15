using System.Text.Json;
using System.Text.Json.Serialization;
using Kairos103.Core.Data;
using Kairos103.Core.Evaluation;

namespace Kairos103.Core.Models;

/// <summary>
/// Metric summary of model evaluation over a dataset.
/// </summary>
public sealed record EvaluationResult(
    double Accuracy,
    int TotalSamples,
    int CorrectSamples,
    int[,] ConfusionMatrix,
    double AvgSpikesPerSample,
    double AvgFrontEndMacsPerSample,
    double AvgAccumulateOpsPerSample);

/// <summary>
/// Serializable model state container.
/// </summary>
public sealed class KairosModelMetadata
{
    public string Architecture { get; set; } = "Kairos103";
    public string Version { get; set; } = "1.1.0";
    public int SubregionCount { get; set; }
    public int Seed { get; set; }
    public int ClassCount { get; set; }
    public int TotalFeatures { get; set; }
    public int HiddenDim { get; set; } = 32;
    public double[][] Weights { get; set; } = [];
    public double[] Bias { get; set; } = [];
    public double[][] W1 { get; set; } = [];
    public double[] B1 { get; set; } = [];
    public double[][] W2 { get; set; } = [];
    public double[] B2 { get; set; } = [];
    public double[] Thresholds { get; set; } = [];
}

/// <summary>
/// High-level "black-box" classifier interface for Kairos103.
/// Encapsulates the event-driven front-end, modular LIF subregions, and margin-based rank readout.
/// </summary>
public sealed class KairosClassifier
{
    public int SubregionCount { get; private set; }
    public int Seed { get; private set; }
    public int ClassCount { get; private set; }
    public bool IsTrained { get; private set; }
    public double[] Thresholds { get; set; } = [];

    public int TotalNeurons => _network?.TotalNeurons ?? (SubregionCount * 4);
    public int TotalFeatures => _trainer?.FeatureDim ?? _network?.TotalFeatures ?? (SubregionCount * 24);
    public int HiddenDim => _trainer?.HiddenDim ?? 0;

    private KairosNetwork _network;
    private PairwiseTrainer? _trainer;

    public KairosClassifier(int subregionCount = 16, int seed = 20261010)
    {
        SubregionCount = subregionCount;
        Seed = seed;
        _network = new KairosNetwork(subregionCount, seed);
    }

    /// <summary>
    /// <summary>
    /// Fits the model on a pre-loaded list of SignalSample objects with optional class weighting.
    /// </summary>
    public void Fit(
        IReadOnlyList<SignalSample> samples,
        int epochs = 25,
        double lr = 0.05,
        int hiddenDim = 32,
        Action<int, int>? onProgress = null,
        double[]? classWeights = null)
    {
        if (samples == null || samples.Count == 0)
            throw new ArgumentException("Dataset cannot be null or empty.", nameof(samples));

        int maxLabel = samples.Max(s => s.Label);
        int uniqueLabels = samples.Select(s => s.Label).Distinct().Count();
        ClassCount = Math.Max(maxLabel + 1, uniqueLabels);

        int auxCount = 0;
        if (samples.Count > 0 && samples[0].Values.Length > samples[0].EffectiveLength && samples[0].EffectiveLength > 0)
        {
            auxCount = samples[0].Values.Length - samples[0].EffectiveLength;
        }

        int totalInputDim = _network.TotalFeatures + auxCount;
        _trainer = new PairwiseTrainer(ClassCount, totalInputDim, hiddenDim: hiddenDim, seed: Seed);

        var encodedTrain = EncodeSamples(samples, auxCount, totalInputDim);
        _trainer.Train(encodedTrain, epochs: epochs, lr: lr, onProgress: onProgress, classWeights: classWeights);
        IsTrained = true;
    }

    /// <summary>
    /// Fits the model while evaluating validation Macro F1 at each epoch, automatically checkpointing the best weights.
    /// </summary>
    public MacroF1Result FitWithValidation(
        IReadOnlyList<SignalSample> trainSamples,
        IReadOnlyList<SignalSample> valSamples,
        int epochs = 30,
        double lr = 0.02,
        int hiddenDim = 32,
        double[]? classWeights = null,
        Action<int, int, double, double>? onEpochCompleted = null,
        bool calibrateThresholds = true)
    {
        if (trainSamples == null || trainSamples.Count == 0)
            throw new ArgumentException("Train samples cannot be empty.", nameof(trainSamples));
        if (valSamples == null || valSamples.Count == 0)
            throw new ArgumentException("Validation samples cannot be empty.", nameof(valSamples));

        int maxLabel = Math.Max(trainSamples.Max(s => s.Label), valSamples.Max(s => s.Label));
        int uniqueLabels = trainSamples.Select(s => s.Label).Union(valSamples.Select(s => s.Label)).Distinct().Count();
        ClassCount = Math.Max(maxLabel + 1, uniqueLabels);

        int auxCount = 0;
        if (trainSamples.Count > 0 && trainSamples[0].Values.Length > trainSamples[0].EffectiveLength && trainSamples[0].EffectiveLength > 0)
        {
            auxCount = trainSamples[0].Values.Length - trainSamples[0].EffectiveLength;
        }

        int totalInputDim = _network.TotalFeatures + auxCount;
        _trainer = new PairwiseTrainer(ClassCount, totalInputDim, hiddenDim: hiddenDim, seed: Seed);

        Console.WriteLine($"\n[1/2] Converting continuous signals to neuromorphic rank-order spike features...");
        Console.Out.Flush();
        var encodedTrain = EncodeSamples(trainSamples, auxCount, totalInputDim, "train");
        var encodedVal = EncodeSamples(valSamples, auxCount, totalInputDim, "val");
        Console.WriteLine($"[1/2] Feature extraction complete. Feature vector dimension: {totalInputDim}\n");
        Console.WriteLine($"[2/2] Training readout layer ({epochs} epochs, Adam optimizer)...");
        Console.Out.Flush();

        var valActuals = valSamples.Select(s => s.Label).ToArray();
        var valPreds = new int[valSamples.Count];

        PairwiseTrainer? bestTrainer = null;
        double bestMacroF1 = -1.0;
        MacroF1Result? bestResult = null;

        for (int ep = 0; ep < epochs; ep++)
        {
            _trainer.TrainEpoch(encodedTrain, lr: lr, classWeights: classWeights);

            // Predict on validation set
            int correctVal = 0;
            for (int i = 0; i < encodedVal.Count; i++)
            {
                int p = _trainer.Predict(encodedVal[i].Features).Prediction;
                valPreds[i] = p;
                if (p == valActuals[i]) correctVal++;
            }

            double valAcc = (double)correctVal / encodedVal.Count;
            var currentMetrics = MacroF1Metrics.Compute(valActuals, valPreds, ClassCount);

            if (currentMetrics.MacroF1 > bestMacroF1)
            {
                bestMacroF1 = currentMetrics.MacroF1;
                bestTrainer = _trainer.Clone();
                bestResult = currentMetrics;
            }

            onEpochCompleted?.Invoke(ep + 1, epochs, valAcc, currentMetrics.MacroF1);
            Console.Out.Flush();
        }

        if (bestTrainer != null)
        {
            _trainer = bestTrainer;
        }

        IsTrained = true;

        if (calibrateThresholds && encodedVal.Count > 0)
        {
            Console.WriteLine("\n[3/3] Calibrating decision thresholds on validation fold to maximize Macro F1...");
            var calResult = CalibrateThresholds(encodedVal);
            Console.WriteLine($"  Optimal Thresholds: [{string.Join(", ", Thresholds.Select((t, c) => $"Class {c}={t:F2}"))}]");
            Console.WriteLine($"  Calibrated Validation Macro F1: {calResult.MacroF1:F4} (Accuracy: {calResult.Accuracy * 100:F2}%)\n");
            Console.Out.Flush();
            return calResult;
        }

        return bestResult ?? MacroF1Metrics.Compute(valActuals, valPreds, ClassCount);
    }

    /// <summary>
    /// Calibrates class decision thresholds on validation probabilities using coordinate search to maximize Macro F1.
    /// </summary>
    public MacroF1Result CalibrateThresholds(IReadOnlyList<(double[] Features, int Label)> valEncoded)
    {
        EnsureTrained();
        int n = valEncoded.Count;
        double[][] probs = new double[n][];
        int[] actuals = new int[n];
        for (int i = 0; i < n; i++)
        {
            probs[i] = _trainer!.PredictProbabilities(valEncoded[i].Features);
            actuals[i] = valEncoded[i].Label;
        }

        Thresholds = Enumerable.Repeat(1.0, ClassCount).ToArray();
        var baseRes = EvaluateWithThresholds(probs, actuals, Thresholds);
        double bestMacroF1 = baseRes.MacroF1;
        double[] bestThresh = (double[])Thresholds.Clone();

        double[] candidates = [0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.25, 1.50, 1.80, 2.20, 2.60, 3.0, 3.5];

        // Coordinate search over class thresholds
        for (int iter = 0; iter < 4; iter++)
        {
            for (int c = 0; c < ClassCount; c++)
            {
                double bestCand = bestThresh[c];
                foreach (double cand in candidates)
                {
                    bestThresh[c] = cand;
                    var res = EvaluateWithThresholds(probs, actuals, bestThresh);
                    if (res.MacroF1 > bestMacroF1)
                    {
                        bestMacroF1 = res.MacroF1;
                        bestCand = cand;
                    }
                }
                bestThresh[c] = bestCand;
            }
        }

        Thresholds = bestThresh;
        return EvaluateWithThresholds(probs, actuals, Thresholds);
    }

    private MacroF1Result EvaluateWithThresholds(double[][] probs, int[] actuals, double[] thresh)
    {
        int[] preds = new int[actuals.Length];
        for (int i = 0; i < actuals.Length; i++)
        {
            int best = 0;
            double bestScore = probs[i][0] / Math.Max(1e-4, thresh[0]);
            for (int c = 1; c < ClassCount; c++)
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
        return MacroF1Metrics.Compute(actuals, preds, ClassCount);
    }

    private List<(double[] Features, int Label)> EncodeSamples(
        IReadOnlyList<SignalSample> samples,
        int auxCount,
        int totalInputDim,
        string phaseName = "signals")
    {
        var encoded = new List<(double[] Features, int Label)>(samples.Count);
        int reportStep = Math.Max(1000, samples.Count / 10);

        for (int i = 0; i < samples.Count; i++)
        {
            if (samples.Count >= 2000 && (i % reportStep == 0 || i == samples.Count - 1))
            {
                double pct = (double)(i + 1) / samples.Count * 100.0;
                Console.WriteLine($"  [Feature Extraction: {phaseName}] Processed {i + 1:N0} / {samples.Count:N0} ({pct:F1}%)...");
                Console.Out.Flush();
            }

            var s = samples[i];
            var (feats, _) = _network.Forward(s.Values, s.EffectiveLength);

            if (auxCount > 0)
            {
                double[] combined = new double[totalInputDim];
                Array.Copy(feats, combined, _network.TotalFeatures);
                int eff = s.EffectiveLength;
                for (int a = 0; a < auxCount; a++)
                {
                    combined[_network.TotalFeatures + a] = s.Values[eff + a];
                }
                encoded.Add((combined, s.Label));
            }
            else
            {
                encoded.Add((feats, s.Label));
            }
        }
        return encoded;
    }

    /// <summary>
    /// Fits the model from parallel raw signals and integer labels.
    /// </summary>
    public void Fit(
        IReadOnlyList<double[]> signals,
        IReadOnlyList<int> labels,
        int epochs = 25,
        double lr = 0.05,
        int hiddenDim = 32,
        Action<int, int>? onProgress = null)
    {
        if (signals.Count != labels.Count)
            throw new ArgumentException("Signals and labels must have the same count.");

        var samples = new List<SignalSample>(signals.Count);
        for (int i = 0; i < signals.Count; i++)
        {
            samples.Add(new SignalSample(labels[i], signals[i], signals[i].Length));
        }

        Fit(samples, epochs, lr, hiddenDim, onProgress);
    }

    /// <summary>
    /// Fits the model directly from a CSV file.
    /// </summary>
    public void Fit(
        string csvPath,
        int epochs = 25,
        double lr = 0.05,
        int hiddenDim = 32,
        Action<int, int>? onProgress = null)
    {
        var samples = SignalDataset.LoadCsv(csvPath);
        Fit(samples, epochs, lr, hiddenDim, onProgress);
    }

    /// <summary>
    /// Predicts the class of a continuous 1D signal with optional auxiliary features.
    /// Applies calibrated thresholds if available.
    /// </summary>
    public int Predict(ReadOnlySpan<double> signal, int effectiveLength = -1)
    {
        if (Thresholds != null && Thresholds.Length == ClassCount)
        {
            double[] probs = PredictProbabilities(signal, effectiveLength);
            int best = 0;
            double bestScore = probs[0] / Math.Max(1e-4, Thresholds[0]);
            for (int c = 1; c < ClassCount; c++)
            {
                double s = probs[c] / Math.Max(1e-4, Thresholds[c]);
                if (s > bestScore)
                {
                    bestScore = s;
                    best = c;
                }
            }
            return best;
        }

        var (pred, _, _, _, _) = PredictDetailed(signal, effectiveLength);
        return pred;
    }

    /// <summary>
    /// Computes predicted class probabilities for a 1D signal with optional auxiliary features.
    /// </summary>
    public double[] PredictProbabilities(ReadOnlySpan<double> signal, int effectiveLength = -1)
    {
        EnsureTrained();
        int len = effectiveLength > 0 ? effectiveLength : signal.Length;
        var (feats, _) = _network.Forward(signal, len);

        double[] inputFeats = feats;
        int auxCount = signal.Length - len;
        if (auxCount > 0 && _trainer!.FeatureDim == _network.TotalFeatures + auxCount)
        {
            inputFeats = new double[_trainer.FeatureDim];
            Array.Copy(feats, inputFeats, _network.TotalFeatures);
            for (int a = 0; a < auxCount; a++)
            {
                inputFeats[_network.TotalFeatures + a] = signal[len + a];
            }
        }

        return _trainer!.PredictProbabilities(inputFeats);
    }

    /// <summary>
    /// Detailed prediction including spike tally and operations count.
    /// </summary>
    public (int Prediction, int TotalSpikes, long FrontEndMacs, int AccumulateOps, double[] Features) PredictDetailed(
        ReadOnlySpan<double> signal,
        int effectiveLength = -1)
    {
        EnsureTrained();
        int len = effectiveLength > 0 ? effectiveLength : signal.Length;
        var (feats, spikes, macs) = _network.ForwardDetailed(signal, len);

        double[] inputFeats = feats;
        int auxCount = signal.Length - len;
        if (auxCount > 0 && _trainer!.FeatureDim == _network.TotalFeatures + auxCount)
        {
            inputFeats = new double[_trainer.FeatureDim];
            Array.Copy(feats, inputFeats, _network.TotalFeatures);
            for (int a = 0; a < auxCount; a++)
            {
                inputFeats[_network.TotalFeatures + a] = signal[len + a];
            }
        }

        var (pred, accOps) = _trainer!.Predict(inputFeats);
        return (pred, spikes, macs, accOps, inputFeats);
    }

    /// <summary>
    /// Predicts classes for a collection of signals in batch mode.
    /// </summary>
    public int[] PredictBatch(IReadOnlyList<double[]> signals)
    {
        EnsureTrained();
        int[] results = new int[signals.Count];
        for (int i = 0; i < signals.Count; i++)
        {
            results[i] = Predict(signals[i]);
        }
        return results;
    }

    /// <summary>
    /// Evaluates model accuracy and operational efficiency over a test dataset.
    /// </summary>
    public EvaluationResult Evaluate(IReadOnlyList<SignalSample> samples)
    {
        EnsureTrained();
        if (samples == null || samples.Count == 0)
            throw new ArgumentException("Samples cannot be empty.");

        int total = samples.Count;
        int correct = 0;
        int[,] cm = new int[ClassCount, ClassCount];

        long totalSpikes = 0;
        long totalMacs = 0;
        long totalOps = 0;

        foreach (var sample in samples)
        {
            var (pred, spikes, macs, accOps, _) = PredictDetailed(sample.Values, sample.EffectiveLength);
            totalSpikes += spikes;
            totalMacs += macs;
            totalOps += accOps;

            if (sample.Label < ClassCount && pred < ClassCount)
            {
                cm[sample.Label, pred]++;
            }

            if (pred == sample.Label)
            {
                correct++;
            }
        }

        double accuracy = (double)correct / total;
        return new EvaluationResult(
            Accuracy: accuracy,
            TotalSamples: total,
            CorrectSamples: correct,
            ConfusionMatrix: cm,
            AvgSpikesPerSample: (double)totalSpikes / total,
            AvgFrontEndMacsPerSample: (double)totalMacs / total,
            AvgAccumulateOpsPerSample: (double)totalOps / total);
    }

    /// <summary>
    /// Serializes and saves the trained model to disk.
    /// </summary>
    public void Save(string filePath)
    {
        EnsureTrained();
        var meta = new KairosModelMetadata
        {
            SubregionCount = SubregionCount,
            Seed = Seed,
            ClassCount = ClassCount,
            TotalFeatures = _trainer!.FeatureDim,
            HiddenDim = _trainer.HiddenDim,
            Weights = _trainer.Weights,
            Bias = _trainer.Bias,
            W1 = _trainer.W1,
            B1 = _trainer.B1,
            W2 = _trainer.W2,
            B2 = _trainer.B2,
            Thresholds = Thresholds
        };

        var options = new JsonSerializerOptions { WriteIndented = true };
        string json = JsonSerializer.Serialize(meta, options);
        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Loads a trained Kairos103 model from disk.
    /// </summary>
    public static KairosClassifier Load(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);

        string json = File.ReadAllText(filePath);
        var meta = JsonSerializer.Deserialize<KairosModelMetadata>(json)
                   ?? throw new InvalidOperationException("Failed to deserialize model metadata.");

        var classifier = new KairosClassifier(meta.SubregionCount, meta.Seed)
        {
            ClassCount = meta.ClassCount,
            IsTrained = true,
            Thresholds = meta.Thresholds ?? []
        };

        classifier._trainer = new PairwiseTrainer(meta.ClassCount, meta.TotalFeatures, meta.HiddenDim, meta.Seed);

        if (meta.HiddenDim > 0 && meta.W1 != null && meta.W1.Length > 0)
        {
            for (int j = 0; j < meta.HiddenDim; j++)
            {
                Array.Copy(meta.W1[j], classifier._trainer.W1[j], meta.TotalFeatures);
            }
            if (meta.B1 != null) Array.Copy(meta.B1, classifier._trainer.B1, meta.HiddenDim);

            for (int c = 0; c < meta.ClassCount; c++)
            {
                Array.Copy(meta.W2[c], classifier._trainer.W2[c], meta.HiddenDim);
            }
            if (meta.B2 != null) Array.Copy(meta.B2, classifier._trainer.B2, meta.ClassCount);
        }
        else if (meta.Weights != null && meta.Weights.Length > 0)
        {
            for (int c = 0; c < meta.ClassCount; c++)
            {
                Array.Copy(meta.Weights[c], classifier._trainer.Weights[c], meta.TotalFeatures);
            }
            if (meta.Bias != null && meta.Bias.Length == meta.ClassCount)
            {
                Array.Copy(meta.Bias, classifier._trainer.Bias, meta.ClassCount);
            }
        }

        return classifier;
    }

    private void EnsureTrained()
    {
        if (!IsTrained || _trainer == null)
            throw new InvalidOperationException("The model has not been trained yet. Call Fit() or Load() first.");
    }
}
