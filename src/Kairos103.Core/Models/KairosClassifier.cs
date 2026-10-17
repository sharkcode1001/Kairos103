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
    public int AuxVersion { get; set; } = 3;
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
    public double[] DecisionBias { get; set; } = [];
    public double[] CalibScale { get; set; } = [];
    public double[] CalibBias { get; set; } = [];
    public bool IsOvr { get; set; } = false;
    public double[][][] OvrW1 { get; set; } = [];
    public double[][] OvrB1 { get; set; } = [];
    public double[][] OvrW2 { get; set; } = [];
    public double[] OvrB2 { get; set; } = [];
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

    /// <summary>
    /// Additive per-class biases applied on calibrated logits at decision time: argmax_c(logit_c + Bias_c).
    /// When empty, the legacy probability/threshold decision rule is used instead.
    /// </summary>
    public double[] DecisionBias { get; set; } = [];
    public double[] CalibScale
    {
        get => _ovrTrainer?.CalibScale ?? _calibScale;
        set
        {
            _calibScale = value;
            if (_ovrTrainer != null) _ovrTrainer.CalibScale = value;
        }
    }
    private double[] _calibScale = [];

    public double[] CalibBias
    {
        get => _ovrTrainer?.CalibBias ?? _calibBias;
        set
        {
            _calibBias = value;
            if (_ovrTrainer != null) _ovrTrainer.CalibBias = value;
        }
    }
    private double[] _calibBias = [];

    public bool IsOvr => _ovrTrainer != null;

    public int TotalNeurons => _network?.TotalNeurons ?? (SubregionCount * 4);
    public int TotalFeatures => _ovrTrainer?.FeatureDim ?? _trainer?.FeatureDim ?? _network?.TotalFeatures ?? (SubregionCount * 24);
    public int HiddenDim => _ovrTrainer?.HiddenDim ?? _trainer?.HiddenDim ?? 0;

    private KairosNetwork _network;
    private PairwiseTrainer? _trainer;
    private OvrTrainer? _ovrTrainer;

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
    /// Fits the model on 100% of the training dataset without validation holdout, using pre-validated optimal thresholds.
    /// </summary>
    public void FitFull(
        IReadOnlyList<SignalSample> trainSamples,
        int epochs = 28,
        double lr = 0.02,
        int hiddenDim = 64,
        double[]? classWeights = null,
        double[]? defaultThresholds = null,
        Action<int, int>? onProgress = null)
    {
        if (trainSamples == null || trainSamples.Count == 0)
            throw new ArgumentException("Train samples cannot be empty.", nameof(trainSamples));

        int maxLabel = trainSamples.Max(s => s.Label);
        int uniqueLabels = trainSamples.Select(s => s.Label).Distinct().Count();
        ClassCount = Math.Max(maxLabel + 1, uniqueLabels);

        int auxCount = 0;
        if (trainSamples.Count > 0 && trainSamples[0].Values.Length > trainSamples[0].EffectiveLength && trainSamples[0].EffectiveLength > 0)
        {
            auxCount = trainSamples[0].Values.Length - trainSamples[0].EffectiveLength;
        }

        int totalInputDim = _network.TotalFeatures + auxCount;
        _trainer = new PairwiseTrainer(ClassCount, totalInputDim, hiddenDim: hiddenDim, seed: Seed);

        Console.WriteLine($"\n[1/2] Converting continuous signals to neuromorphic rank-order spike features (Full Dataset: {trainSamples.Count:N0} samples)...");
        Console.Out.Flush();
        var encodedTrain = EncodeSamples(trainSamples, auxCount, totalInputDim, "full-train");
        Console.WriteLine($"[1/2] Feature extraction complete. Feature vector dimension: {totalInputDim}\n");
        Console.WriteLine($"[2/2] Training readout layer on 100% data ({epochs} epochs, Adam optimizer)...");
        Console.Out.Flush();

        _trainer.Train(encodedTrain, epochs: epochs, lr: lr, onProgress: onProgress, classWeights: classWeights);
        IsTrained = true;

        Thresholds = (defaultThresholds != null && defaultThresholds.Length == ClassCount)
            ? defaultThresholds
            : DefaultThresholds(ClassCount, [0.50, 0.18, 0.90, 1.00]);
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
            Console.WriteLine("\n[3/3] Calibrating logits (Platt) & decision thresholds on validation fold to maximize Macro F1...");
            var calResult = CalibrateThresholds(encodedVal);
            if (CalibScale != null && CalibScale.Length == ClassCount)
            {
                Console.WriteLine($"  Platt Scaling: [{string.Join(", ", CalibScale.Select((a, c) => $"Class {c}: a={a:F3}, b={CalibBias[c]:F3}"))}]");
            }
            Console.WriteLine($"  Optimal Thresholds: [{string.Join(", ", Thresholds.Select((t, c) => $"Class {c}={t:F2}"))}]");
            Console.WriteLine($"  Calibrated Validation Macro F1: {calResult.MacroF1:F4} (Accuracy: {calResult.Accuracy * 100:F2}%)");
            if (TryRejectCalibration(calResult, bestResult))
            {
                return bestResult!;
            }
            Console.WriteLine();
            Console.Out.Flush();
            return calResult;
        }

        return bestResult ?? MacroF1Metrics.Compute(valActuals, valPreds, ClassCount);
    }

    /// <summary>
    /// Fits 1D Platt scaling (a_c * z_c + b_c) per class head on validation logits using Newton-Raphson.
    /// Regularizes binary targets using Lin, Lin & Weng (2007) formulation to prevent overfitting/divergence.
    /// </summary>
    public void FitPlattCalibration(double[][] logits, int[] actuals)
    {
        int n = logits.Length;
        if (n == 0) return;

        double[] scales = new double[ClassCount];
        double[] biases = new double[ClassCount];

        for (int c = 0; c < ClassCount; c++)
        {
            int nPos = 0;
            for (int i = 0; i < n; i++)
            {
                if (actuals[i] == c) nPos++;
            }
            int nNeg = n - nPos;

            // Target regularization (Lin et al. 2007)
            double tPos = (nPos + 1.0) / (nPos + 2.0);
            double tNeg = 1.0 / (nNeg + 2.0);

            // Initial parameters: scale a = 1.0, bias b = 0.0
            double a = 1.0;
            double b = 0.0;
            double lambda = 1e-4;

            // Newton-Raphson optimization
            for (int iter = 0; iter < 25; iter++)
            {
                double gradA = 0.0;
                double gradB = 0.0;
                double hAA = 0.0;
                double hAB = 0.0;
                double hBB = 0.0;

                for (int i = 0; i < n; i++)
                {
                    double z = logits[i][c];
                    double target = (actuals[i] == c) ? tPos : tNeg;
                    double p = Sigmoid(a * z + b);
                    p = Math.Clamp(p, 1e-12, 1.0 - 1e-12);

                    double diff = p - target;
                    gradA += diff * z;
                    gradB += diff;

                    double w = p * (1.0 - p);
                    hAA += w * z * z;
                    hAB += w * z;
                    hBB += w;
                }

                hAA += lambda;
                hBB += lambda;

                double det = (hAA * hBB) - (hAB * hAB);
                if (Math.Abs(det) < 1e-12)
                {
                    break;
                }

                double deltaA = -(hBB * gradA - hAB * gradB) / det;
                double deltaB = -(-hAB * gradA + hAA * gradB) / det;

                // Step damping to prevent oscillation
                deltaA = Math.Clamp(deltaA, -2.0, 2.0);
                deltaB = Math.Clamp(deltaB, -2.0, 2.0);

                a += deltaA;
                b += deltaB;

                // Monotonicity guard: logit should remain positively correlated with class probability
                if (a < 0.01) a = 0.01;

                if (Math.Abs(deltaA) < 1e-5 && Math.Abs(deltaB) < 1e-5)
                {
                    break;
                }
            }

            scales[c] = a;
            biases[c] = b;
        }

        CalibScale = scales;
        CalibBias = biases;
    }

    /// <summary>
    /// Calibrates class decision thresholds and Platt scaling parameters on validation logits using coordinate search to maximize Macro F1.
    /// </summary>
    public MacroF1Result CalibrateThresholds(IReadOnlyList<(double[] Features, int Label)> valEncoded)
    {
        EnsureTrained();
        int n = valEncoded.Count;
        double[][] probs = new double[n][];
        int[] actuals = new int[n];

        if (_ovrTrainer != null)
        {
            double[][] logits = new double[n][];
            for (int i = 0; i < n; i++)
            {
                logits[i] = _ovrTrainer.PredictLogits(valEncoded[i].Features);
                actuals[i] = valEncoded[i].Label;
            }

            // 1. Fit Platt scaling calibration on raw OvR validation logits
            FitPlattCalibration(logits, actuals);

            // 2. Compute calibrated probabilities using the newly fitted parameters
            for (int i = 0; i < n; i++)
            {
                probs[i] = _ovrTrainer.PredictProbabilities(valEncoded[i].Features);
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                probs[i] = _trainer!.PredictProbabilities(valEncoded[i].Features);
                actuals[i] = valEncoded[i].Label;
            }
        }

        Thresholds = Enumerable.Repeat(1.0, ClassCount).ToArray();
        var baseRes = EvaluateWithThresholds(probs, actuals, Thresholds);
        double bestMacroF1 = baseRes.MacroF1;
        double[] bestThresh = (double[])Thresholds.Clone();

        double[] candidates = [0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 1.0, 1.10, 1.25, 1.50, 1.80, 2.20, 2.60, 3.0, 3.5];

        // 3. Coordinate search over class thresholds to directly maximize Macro F1
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

    /// <summary>
    /// Rejects a calibration that lowered validation Macro F1 versus the raw argmax checkpoint,
    /// restoring the raw decision rule (identity thresholds, no Platt scaling).
    /// Returns true when calibration was rejected.
    /// </summary>
    private bool TryRejectCalibration(MacroF1Result calibrated, MacroF1Result? rawBest)
    {
        if (rawBest == null || calibrated.MacroF1 >= rawBest.MacroF1)
        {
            return false;
        }

        CalibScale = [];
        CalibBias = [];
        Thresholds = Enumerable.Repeat(1.0, ClassCount).ToArray();
        Console.WriteLine($"  [Calibration rejected] Raw argmax Macro F1 {rawBest.MacroF1:F4} >= calibrated {calibrated.MacroF1:F4}; keeping raw decision rule.\n");
        Console.Out.Flush();
        return true;
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

    /// <summary>
    /// Fits the model using One-vs-Rest (OvR) binary heads on shared neuromorphic features with validation monitoring.
    /// </summary>
    public MacroF1Result FitOvrWithValidation(
        IReadOnlyList<SignalSample> trainSamples,
        IReadOnlyList<SignalSample> valSamples,
        int epochs = 30,
        double lr = 0.02,
        int hiddenDim = 32,
        double weightPower = 0.5,
        Action<int, int, double, double>? onEpochCompleted = null,
        bool calibrateThresholds = true,
        double[]? weightPowers = null)
    {
        if (trainSamples == null || trainSamples.Count == 0)
            throw new ArgumentException("Train samples cannot be empty.", nameof(trainSamples));
        if (valSamples == null || valSamples.Count == 0)
            throw new ArgumentException("Validation samples cannot be empty.", nameof(valSamples));

        int auxCount = 0;
        if (trainSamples.Count > 0 && trainSamples[0].Values.Length > trainSamples[0].EffectiveLength && trainSamples[0].EffectiveLength > 0)
        {
            auxCount = trainSamples[0].Values.Length - trainSamples[0].EffectiveLength;
        }

        int totalInputDim = _network.TotalFeatures + auxCount;
        Console.WriteLine($"\n[1/2] Converting continuous signals to neuromorphic rank-order spike features (OvR Architecture)...");
        Console.Out.Flush();
        var encodedTrain = EncodeSamples(trainSamples, auxCount, totalInputDim, "train");
        var encodedVal = EncodeSamples(valSamples, auxCount, totalInputDim, "val");
        Console.WriteLine($"[1/2] Feature extraction complete. Feature vector dimension: {totalInputDim}\n");

        return FitOvrWithValidation(
            encodedTrain,
            encodedVal,
            epochs: epochs,
            lr: lr,
            hiddenDim: hiddenDim,
            weightPower: weightPower,
            onEpochCompleted: onEpochCompleted,
            calibrateThresholds: calibrateThresholds,
            weightPowers: weightPowers);
    }

    /// <summary>
    /// Fits the OvR model directly from pre-encoded feature vectors, avoiding redundant neuromorphic front-end re-computation.
    /// </summary>
    public MacroF1Result FitOvrWithValidation(
        IReadOnlyList<(double[] Features, int Label)> encodedTrain,
        IReadOnlyList<(double[] Features, int Label)> encodedVal,
        int epochs = 30,
        double lr = 0.02,
        int hiddenDim = 32,
        double weightPower = 0.5,
        Action<int, int, double, double>? onEpochCompleted = null,
        bool calibrateThresholds = true,
        double[]? weightPowers = null)
    {
        if (encodedTrain == null || encodedTrain.Count == 0)
            throw new ArgumentException("Encoded train samples cannot be empty.", nameof(encodedTrain));
        if (encodedVal == null || encodedVal.Count == 0)
            throw new ArgumentException("Encoded val samples cannot be empty.", nameof(encodedVal));

        int maxLabel = Math.Max(encodedTrain.Max(s => s.Label), encodedVal.Max(s => s.Label));
        int uniqueLabels = encodedTrain.Select(s => s.Label).Union(encodedVal.Select(s => s.Label)).Distinct().Count();
        ClassCount = Math.Max(maxLabel + 1, uniqueLabels);

        int totalInputDim = encodedTrain[0].Features.Length;
        _ovrTrainer = new OvrTrainer(ClassCount, totalInputDim, hiddenDim: hiddenDim, seed: Seed);
        _trainer = null;
        Thresholds = Enumerable.Repeat(1.0, ClassCount).ToArray();

        Console.WriteLine($"[2/2] Training {ClassCount} independent OvR binary heads ({epochs} epochs, Adam optimizer)...");
        Console.Out.Flush();

        var valActuals = encodedVal.Select(s => s.Label).ToArray();
        var valPreds = new int[encodedVal.Count];

        int nTrain = encodedTrain.Count;
        double[] posWeights = new double[ClassCount];
        for (int c = 0; c < ClassCount; c++)
        {
            int pos = 0;
            for (int i = 0; i < nTrain; i++)
            {
                if (encodedTrain[i].Label == c) pos++;
            }
            int neg = nTrain - pos;
            double wp = (weightPowers != null && c < weightPowers.Length) ? weightPowers[c] : weightPower;
            posWeights[c] = pos > 0 ? Math.Pow((double)neg / pos, wp) : 1.0;
        }

        MacroF1Result? bestResult = null;
        double bestValF1 = -1.0;
        OvrTrainer? bestCheckpoint = null;
        var shuffleRng = new Random(Seed);
        for (int ep = 1; ep <= epochs; ep++)
        {
            double currentLr = lr * (0.1 + (0.9 * 0.5 * (1.0 + Math.Cos(Math.PI * (ep - 1) / Math.Max(1, epochs)))));
            _ovrTrainer.TrainEpoch(encodedTrain, posWeights, lr: currentLr, rng: shuffleRng);

            for (int i = 0; i < encodedVal.Count; i++)
            {
                var (pred, _) = _ovrTrainer.Predict(encodedVal[i].Features, Thresholds);
                valPreds[i] = pred;
            }

            var epochResult = MacroF1Metrics.Compute(valActuals, valPreds, ClassCount);
            onEpochCompleted?.Invoke(ep, epochs, epochResult.Accuracy, epochResult.MacroF1);

            if (epochResult.MacroF1 > bestValF1)
            {
                bestValF1 = epochResult.MacroF1;
                bestResult = epochResult;
                bestCheckpoint = _ovrTrainer.Clone();
            }
        }

        if (bestCheckpoint != null)
        {
            _ovrTrainer = bestCheckpoint;
        }

        IsTrained = true;

        if (calibrateThresholds && encodedVal.Count > 0)
        {
            Console.WriteLine("\n[3/3] Calibrating logits (Platt) & decision thresholds across OvR heads to maximize Macro F1...");
            var calResult = CalibrateThresholds(encodedVal);
            if (CalibScale != null && CalibScale.Length == ClassCount)
            {
                Console.WriteLine($"  Platt Scaling: [{string.Join(", ", CalibScale.Select((a, c) => $"Class {c}: a={a:F3}, b={CalibBias[c]:F3}"))}]");
            }
            Console.WriteLine($"  Optimal Thresholds: [{string.Join(", ", Thresholds.Select((t, c) => $"Class {c}={t:F2}"))}]");
            Console.WriteLine($"  Calibrated Validation Macro F1: {calResult.MacroF1:F4} (Accuracy: {calResult.Accuracy * 100:F2}%)");
            if (TryRejectCalibration(calResult, bestResult))
            {
                return bestResult!;
            }
            Console.WriteLine();
            Console.Out.Flush();
            return calResult;
        }

        return bestResult ?? MacroF1Metrics.Compute(valActuals, valPreds, ClassCount);
    }

    /// <summary>
    /// Fits the OvR model on 100% of the training dataset.
    /// </summary>
    public void FitOvrFull(
        IReadOnlyList<SignalSample> trainSamples,
        int epochs = 28,
        double lr = 0.02,
        int hiddenDim = 32,
        double weightPower = 0.5,
        double[]? defaultThresholds = null,
        Action<int, int>? onProgress = null,
        double[]? weightPowers = null)
    {
        if (trainSamples == null || trainSamples.Count == 0)
            throw new ArgumentException("Train samples cannot be empty.", nameof(trainSamples));

        int maxLabel = trainSamples.Max(s => s.Label);
        int uniqueLabels = trainSamples.Select(s => s.Label).Distinct().Count();
        ClassCount = Math.Max(maxLabel + 1, uniqueLabels);

        int auxCount = 0;
        if (trainSamples.Count > 0 && trainSamples[0].Values.Length > trainSamples[0].EffectiveLength && trainSamples[0].EffectiveLength > 0)
        {
            auxCount = trainSamples[0].Values.Length - trainSamples[0].EffectiveLength;
        }

        int totalInputDim = _network.TotalFeatures + auxCount;
        _ovrTrainer = new OvrTrainer(ClassCount, totalInputDim, hiddenDim: hiddenDim, seed: Seed);
        _trainer = null;

        Console.WriteLine($"\n[1/2] Converting continuous signals to neuromorphic rank-order spike features (OvR Full Dataset: {trainSamples.Count:N0} samples)...");
        Console.Out.Flush();
        var encodedTrain = EncodeSamples(trainSamples, auxCount, totalInputDim, "full-train");
        Console.WriteLine($"[1/2] Feature extraction complete. Feature vector dimension: {totalInputDim}\n");
        Console.WriteLine($"[2/2] Training {ClassCount} independent OvR binary heads on 100% data ({epochs} epochs, Adam optimizer)...");
        Console.Out.Flush();

        _ovrTrainer.Train(encodedTrain, epochs: epochs, lr: lr, onProgress: onProgress, weightPower: weightPower, weightPowers: weightPowers);
        IsTrained = true;

        Thresholds = (defaultThresholds != null && defaultThresholds.Length == ClassCount)
            ? defaultThresholds
            : DefaultThresholds(ClassCount, [1.00, 1.10, 1.00, 1.00]);
    }

    /// <summary>
    /// Returns the class-tuned default thresholds when the model's class count matches the tuned preset;
    /// otherwise returns neutral threshold 1.0 for every class so the decision rule remains dimensionally valid.
    /// </summary>
    private static double[] DefaultThresholds(int classCount, double[] tunedPreset)
    {
        return classCount == tunedPreset.Length
            ? (double[])tunedPreset.Clone()
            : Enumerable.Repeat(1.0, classCount).ToArray();
    }

    private IReadOnlyList<(double[] Features, int Label)> EncodeSamples(
        IReadOnlyList<SignalSample> samples,
        int auxCount,
        int totalInputDim,
        string phaseName = "signals")
    {
        var encoded = new (double[] Features, int Label)[samples.Count];
        int total = samples.Count;
        int completed = 0;
        int reportStep = Math.Max(1000, total / 10);

        using var threadLocalNet = new ThreadLocal<KairosNetwork>(() => _network.Clone());

        Parallel.For(0, total, i =>
        {
            var net = threadLocalNet.Value!;
            var s = samples[i];
            var (feats, _) = net.Forward(s.Values, s.EffectiveLength);

            if (auxCount > 0)
            {
                double[] combined = new double[totalInputDim];
                Array.Copy(feats, combined, _network.TotalFeatures);
                int eff = s.EffectiveLength;
                for (int a = 0; a < auxCount; a++)
                {
                    combined[_network.TotalFeatures + a] = s.Values[eff + a];
                }
                encoded[i] = (combined, s.Label);
            }
            else
            {
                encoded[i] = (feats, s.Label);
            }

            if (total >= 2000)
            {
                int currentCompleted = Interlocked.Increment(ref completed);
                if (currentCompleted % reportStep == 0 || currentCompleted == total)
                {
                    double pct = (double)currentCompleted / total * 100.0;
                    Console.WriteLine($"  [Feature Extraction: {phaseName}] Processed {currentCompleted:N0} / {total:N0} ({pct:F1}%)...");
                    Console.Out.Flush();
                }
            }
        });

        return encoded;
    }

    /// <summary>
    /// Converts a collection of signal samples into neuromorphic rank-order features and auxiliary clinical features.
    /// Can be executed once and reused across cross-validation folds.
    /// </summary>
    public IReadOnlyList<(double[] Features, int Label)> EncodeSamples(
        IReadOnlyList<SignalSample> samples,
        string phaseName = "signals")
    {
        if (samples == null || samples.Count == 0) return [];
        int auxCount = 0;
        if (samples.Count > 0 && samples[0].Values.Length > samples[0].EffectiveLength && samples[0].EffectiveLength > 0)
        {
            auxCount = samples[0].Values.Length - samples[0].EffectiveLength;
        }
        int totalInputDim = _network.TotalFeatures + auxCount;
        return EncodeSamples(samples, auxCount, totalInputDim, phaseName);
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
        if (DecisionBias != null && DecisionBias.Length == ClassCount)
        {
            double[] logits = PredictCalibratedLogits(signal, effectiveLength);
            return ArgMaxBias(logits, DecisionBias);
        }

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

        int featDim = _ovrTrainer?.FeatureDim ?? _trainer!.FeatureDim;
        double[] inputFeats = feats;
        int auxCount = signal.Length - len;
        if (auxCount > 0)
        {
            if (featDim != _network.TotalFeatures + auxCount)
            {
                throw new InvalidOperationException(
                    $"Feature dimension mismatch! Model expects {featDim} features " +
                    $"({_network.TotalFeatures} neuromorphic + {featDim - _network.TotalFeatures} aux), " +
                    $"but input signal provided {auxCount} auxiliary features (Total: {_network.TotalFeatures + auxCount}).");
            }

            inputFeats = new double[featDim];
            Array.Copy(feats, inputFeats, _network.TotalFeatures);
            for (int a = 0; a < auxCount; a++)
            {
                inputFeats[_network.TotalFeatures + a] = signal[len + a];
            }
        }

        if (_ovrTrainer != null)
        {
            return _ovrTrainer.PredictProbabilities(inputFeats);
        }
        return _trainer!.PredictProbabilities(inputFeats);
    }

    /// <summary>
    /// Computes raw uncalibrated linear logits for a 1D signal with optional auxiliary features.
    /// </summary>
    public double[] PredictLogits(ReadOnlySpan<double> signal, int effectiveLength = -1)
    {
        EnsureTrained();
        int len = effectiveLength > 0 ? effectiveLength : signal.Length;
        var (feats, _) = _network.Forward(signal, len);

        int featDim = _ovrTrainer?.FeatureDim ?? _trainer!.FeatureDim;
        double[] inputFeats = feats;
        int auxCount = signal.Length - len;
        if (auxCount > 0)
        {
            if (featDim != _network.TotalFeatures + auxCount)
            {
                throw new InvalidOperationException(
                    $"Feature dimension mismatch! Model expects {featDim} features " +
                    $"({_network.TotalFeatures} neuromorphic + {featDim - _network.TotalFeatures} aux), " +
                    $"but input signal provided {auxCount} auxiliary features (Total: {_network.TotalFeatures + auxCount}).");
            }

            inputFeats = new double[featDim];
            Array.Copy(feats, inputFeats, _network.TotalFeatures);
            for (int a = 0; a < auxCount; a++)
            {
                inputFeats[_network.TotalFeatures + a] = signal[len + a];
            }
        }

        if (_ovrTrainer != null)
        {
            return _ovrTrainer.PredictLogits(inputFeats);
        }
        return GetTrainerLogits(inputFeats);
    }

    private static double Sigmoid(double z)
    {
        if (z < -30.0) return 0.0;
        if (z > 30.0) return 1.0;
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    /// <summary>
    /// Computes calibrated decision logits for a 1D signal with optional auxiliary features.
    /// For OvR models this is the Platt-calibrated logit (a_c*z_c + b_c); for softmax models it is the raw logit.
    /// These are the values on which <see cref="DecisionBias"/> is applied.
    /// </summary>
    public double[] PredictCalibratedLogits(ReadOnlySpan<double> signal, int effectiveLength = -1)
    {
        EnsureTrained();
        int len = effectiveLength > 0 ? effectiveLength : signal.Length;
        var (feats, _) = _network.Forward(signal, len);

        int featDim = _ovrTrainer?.FeatureDim ?? _trainer!.FeatureDim;
        double[] inputFeats = feats;
        int auxCount = signal.Length - len;
        if (auxCount > 0)
        {
            if (featDim != _network.TotalFeatures + auxCount)
            {
                throw new InvalidOperationException(
                    $"Feature dimension mismatch! Model expects {featDim} features " +
                    $"({_network.TotalFeatures} neuromorphic + {featDim - _network.TotalFeatures} aux), " +
                    $"but input signal provided {auxCount} auxiliary features (Total: {_network.TotalFeatures + auxCount}).");
            }

            inputFeats = new double[featDim];
            Array.Copy(feats, inputFeats, _network.TotalFeatures);
            for (int a = 0; a < auxCount; a++)
            {
                inputFeats[_network.TotalFeatures + a] = signal[len + a];
            }
        }

        if (_ovrTrainer != null)
        {
            return _ovrTrainer.PredictCalibratedLogits(inputFeats);
        }
        return _trainer!.PredictLogits(inputFeats);
    }

    /// <summary>
    /// Computes calibrated logits directly from a pre-extracted feature vector without running front-end LIF.
    /// </summary>
    public double[] PredictCalibratedLogitsFromFeatures(ReadOnlySpan<double> features)
    {
        EnsureTrained();
        if (_ovrTrainer != null)
        {
            return _ovrTrainer.PredictCalibratedLogits(features);
        }
        return _trainer!.PredictLogits(features.ToArray());
    }

    /// <summary>
    /// Computes class probabilities directly from a pre-extracted feature vector without running front-end LIF.
    /// </summary>
    public double[] PredictProbabilitiesFromFeatures(ReadOnlySpan<double> features)
    {
        EnsureTrained();
        if (_ovrTrainer != null)
        {
            return _ovrTrainer.PredictProbabilities(features);
        }
        return _trainer!.PredictProbabilities(features.ToArray());
    }

    /// <summary>
    /// Returns the index maximizing logit_c + bias_c (class 0 wins ties).
    /// </summary>
    public static int ArgMaxBias(IReadOnlyList<double> logits, IReadOnlyList<double> bias)
    {
        int best = 0;
        double bestScore = logits[0] + bias[0];
        for (int c = 1; c < logits.Count; c++)
        {
            double s = logits[c] + bias[c];
            if (s > bestScore)
            {
                bestScore = s;
                best = c;
            }
        }
        return best;
    }

    /// <summary>
    /// Optimizes additive per-class decision biases on calibrated logits to maximize Macro F1.
    /// Uses coordinate ascent over a coarse-to-fine candidate grid with repeated sweeps.
    /// </summary>
    public static double[] OptimizeDecisionBias(double[][] logits, IReadOnlyList<int> actuals, int classCount, out double bestMacroF1)
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (actuals == null) throw new ArgumentNullException(nameof(actuals));
        if (logits.Length != actuals.Count) throw new ArgumentException("logits and actuals must have equal length.");
        if (classCount < 2) throw new ArgumentOutOfRangeException(nameof(classCount));

        int n = logits.Length;
        int[] preds = new int[n];
        double[] bias = new double[classCount];

        double Evaluate(double[] b)
        {
            for (int i = 0; i < n; i++)
            {
                double[] z = logits[i];
                int best = 0;
                double bestScore = z[0] + b[0];
                for (int c = 1; c < classCount; c++)
                {
                    double s = z[c] + b[c];
                    if (s > bestScore)
                    {
                        bestScore = s;
                        best = c;
                    }
                }
                preds[i] = best;
            }
            return MacroF1Metrics.Compute(actuals, preds, classCount).MacroF1;
        }

        double best = Evaluate(bias);

        double[] coarse = [-3.0, -2.0, -1.5, -1.0, -0.7, -0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0];
        double[] fine = [-0.08, -0.05, -0.03, -0.015, 0.0, 0.015, 0.03, 0.05, 0.08];

        Sweep(bias, coarse, Evaluate, ref best);
        Sweep(bias, fine, Evaluate, ref best);

        bestMacroF1 = Evaluate(bias);
        return bias;
    }

    private static void Sweep(double[] bias, double[] offsets, Func<double[], double> evaluate, ref double best)
    {
        bool improved = true;
        int sweeps = 0;
        while (improved && sweeps < 4)
        {
            improved = false;
            for (int c = 0; c < bias.Length; c++)
            {
                double origin = bias[c];
                double bestVal = origin;
                foreach (double d in offsets)
                {
                    bias[c] = origin + d;
                    double f = evaluate(bias);
                    if (f > best + 1e-9)
                    {
                        best = f;
                        bestVal = bias[c];
                        improved = true;
                    }
                }
                bias[c] = bestVal;
            }
            sweeps++;
        }
    }

    private double[] GetTrainerLogits(double[] features)
    {
        double[] probs = _trainer!.PredictProbabilities(features);
        double[] logits = new double[probs.Length];
        for (int c = 0; c < probs.Length; c++)
        {
            double p = Math.Clamp(probs[c], 1e-7, 1.0 - 1e-7);
            logits[c] = Math.Log(p / (1.0 - p));
        }
        return logits;
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

        int featDim = _ovrTrainer?.FeatureDim ?? _trainer!.FeatureDim;
        double[] inputFeats = feats;
        int auxCount = signal.Length - len;
        if (auxCount > 0 && featDim == _network.TotalFeatures + auxCount)
        {
            inputFeats = new double[featDim];
            Array.Copy(feats, inputFeats, _network.TotalFeatures);
            for (int a = 0; a < auxCount; a++)
            {
                inputFeats[_network.TotalFeatures + a] = signal[len + a];
            }
        }

        if (_ovrTrainer != null)
        {
            int accOps = ClassCount * (_ovrTrainer.HiddenDim * _ovrTrainer.FeatureDim + _ovrTrainer.HiddenDim);
            int pred;
            if (DecisionBias != null && DecisionBias.Length == ClassCount)
            {
                var logits = _ovrTrainer.PredictCalibratedLogits(inputFeats);
                pred = ArgMaxBias(logits, DecisionBias);
            }
            else
            {
                (pred, _) = _ovrTrainer.Predict(inputFeats, Thresholds);
            }
            return (pred, spikes, macs, accOps, inputFeats);
        }
        var (predTr, accOpsTr) = _trainer!.Predict(inputFeats);
        if (DecisionBias != null && DecisionBias.Length == ClassCount)
        {
            var logits = _trainer!.PredictLogits(inputFeats);
            predTr = ArgMaxBias(logits, DecisionBias);
        }
        return (predTr, spikes, macs, accOpsTr, inputFeats);
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
            HiddenDim = HiddenDim,
            Thresholds = Thresholds,
            DecisionBias = DecisionBias,
            CalibScale = CalibScale,
            CalibBias = CalibBias,
            IsOvr = IsOvr,
            AuxVersion = 3
        };

        if (IsOvr && _ovrTrainer != null)
        {
            meta.OvrW1 = _ovrTrainer.W1;
            meta.OvrB1 = _ovrTrainer.B1;
            meta.OvrW2 = _ovrTrainer.W2;
            meta.OvrB2 = _ovrTrainer.B2;
        }
        else if (_trainer != null)
        {
            meta.Weights = _trainer.Weights;
            meta.Bias = _trainer.Bias;
            meta.W1 = _trainer.W1;
            meta.B1 = _trainer.B1;
            meta.W2 = _trainer.W2;
            meta.B2 = _trainer.B2;
        }

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
            Thresholds = meta.Thresholds ?? [],
            DecisionBias = meta.DecisionBias ?? [],
            CalibScale = meta.CalibScale ?? [],
            CalibBias = meta.CalibBias ?? []
        };

        if (meta.AuxVersion < 3)
        {
            Console.WriteLine($"  [WARNING] Model was trained with legacy AuxVersion {meta.AuxVersion}. Runtime uses AuxVersion 3.");
        }

        if (meta.IsOvr && meta.OvrW1 != null && meta.OvrW1.Length > 0)
        {
            var ovr = new OvrTrainer(meta.ClassCount, meta.TotalFeatures, meta.HiddenDim, meta.Seed);
            for (int c = 0; c < meta.ClassCount; c++)
            {
                for (int j = 0; j < meta.HiddenDim; j++)
                {
                    Array.Copy(meta.OvrW1[c][j], ovr.W1[c][j], meta.TotalFeatures);
                }
                Array.Copy(meta.OvrB1[c], ovr.B1[c], meta.HiddenDim);
                Array.Copy(meta.OvrW2[c], ovr.W2[c], meta.HiddenDim);
                ovr.B2[c] = meta.OvrB2[c];
            }
            ovr.CalibScale = classifier.CalibScale;
            ovr.CalibBias = classifier.CalibBias;
            classifier._ovrTrainer = ovr;
        }
        else
        {
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
        }

        return classifier;
    }

    private void EnsureTrained()
    {
        if (!IsTrained || (_trainer == null && _ovrTrainer == null))
            throw new InvalidOperationException("The model has not been trained yet. Call Fit() or Load() first.");
    }
}
