using System.Text.Json;
using System.Text.Json.Serialization;
using Kairos103.Core.Data;

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

    public int TotalNeurons => _network?.TotalNeurons ?? (SubregionCount * 4);
    public int TotalFeatures => _network?.TotalFeatures ?? (SubregionCount * 24);
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
    /// Fits the model on a pre-loaded list of SignalSample objects.
    /// </summary>
    public void Fit(
        IReadOnlyList<SignalSample> samples,
        int epochs = 25,
        double lr = 0.05,
        int hiddenDim = 32,
        Action<int, int>? onProgress = null)
    {
        if (samples == null || samples.Count == 0)
            throw new ArgumentException("Dataset cannot be null or empty.", nameof(samples));

        // Auto-detect classes (find max label + 1 or count unique labels)
        int maxLabel = samples.Max(s => s.Label);
        int uniqueLabels = samples.Select(s => s.Label).Distinct().Count();
        ClassCount = Math.Max(maxLabel + 1, uniqueLabels);

        _trainer = new PairwiseTrainer(ClassCount, _network.TotalFeatures, hiddenDim: hiddenDim, seed: Seed);

        // Pre-encode all signals through the Kairos asynchronous front-end
        var encodedTrain = new List<(double[] Features, int Label)>(samples.Count);
        for (int i = 0; i < samples.Count; i++)
        {
            var s = samples[i];
            var (feats, _) = _network.Forward(s.Values, s.EffectiveLength);
            encodedTrain.Add((feats, s.Label));
        }

        _trainer.Train(encodedTrain, epochs: epochs, lr: lr, onProgress: onProgress);
        IsTrained = true;
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
    /// Predicts the class of a continuous 1D signal.
    /// </summary>
    public int Predict(ReadOnlySpan<double> signal, int effectiveLength = -1)
    {
        EnsureTrained();
        int len = effectiveLength > 0 ? effectiveLength : signal.Length;
        var (feats, _) = _network.Forward(signal, len);
        var (pred, _) = _trainer!.Predict(feats);
        return pred;
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
        var (pred, accOps) = _trainer!.Predict(feats);
        return (pred, spikes, macs, accOps, feats);
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
            TotalFeatures = TotalFeatures,
            HiddenDim = _trainer!.HiddenDim,
            Weights = _trainer.Weights,
            Bias = _trainer.Bias,
            W1 = _trainer.W1,
            B1 = _trainer.B1,
            W2 = _trainer.W2,
            B2 = _trainer.B2
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
            IsTrained = true
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
