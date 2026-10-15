namespace Kairos103.Core.Models;

/// <summary>
/// Supervised One-vs-Rest (OvR) binary readout classifier with shared neuromorphic rank-order features.
/// Trains an independent binary MLP head per class with class-balanced Binary Cross-Entropy loss.
/// Eliminates the softmax bottleneck where dominant classes (Normal, VEB) suppress minority classes (SVEB, Fusion).
/// </summary>
public sealed class OvrTrainer
{
    public int ClassCount { get; }
    public int FeatureDim { get; }
    public int HiddenDim { get; }
    public int Seed { get; }

    // Binary MLP weights per class: [class][hidden][feature]
    public double[][][] W1 { get; }
    public double[][] B1 { get; }
    public double[][] W2 { get; } // [class][hidden]
    public double[] B2 { get; }   // [class]

    // Adam optimizer moments per class
    private readonly double[][][] _mW1, _vW1;
    private readonly double[][] _mB1, _vB1;
    private readonly double[][] _mW2, _vW2;
    private readonly double[] _mB2, _vB2;
    private readonly int[] _adamStep;

    public OvrTrainer(int classCount, int featureDim, int hiddenDim = 32, int seed = 42)
    {
        ClassCount = classCount;
        FeatureDim = featureDim;
        HiddenDim = hiddenDim;
        Seed = seed;

        W1 = new double[classCount][][];
        B1 = new double[classCount][];
        W2 = new double[classCount][];
        B2 = new double[classCount];

        _mW1 = new double[classCount][][];
        _vW1 = new double[classCount][][];
        _mB1 = new double[classCount][];
        _vB1 = new double[classCount][];
        _mW2 = new double[classCount][];
        _vW2 = new double[classCount][];
        _mB2 = new double[classCount];
        _vB2 = new double[classCount];
        _adamStep = new int[classCount];

        var rng = new Random(seed);
        double scale1 = Math.Sqrt(2.0 / featureDim);
        double scale2 = Math.Sqrt(2.0 / hiddenDim);

        for (int c = 0; c < classCount; c++)
        {
            W1[c] = new double[hiddenDim][];
            _mW1[c] = new double[hiddenDim][];
            _vW1[c] = new double[hiddenDim][];
            for (int j = 0; j < hiddenDim; j++)
            {
                W1[c][j] = new double[featureDim];
                _mW1[c][j] = new double[featureDim];
                _vW1[c][j] = new double[featureDim];
                for (int f = 0; f < featureDim; f++)
                {
                    W1[c][j][f] = (rng.NextDouble() * 2.0 - 1.0) * scale1;
                }
            }

            B1[c] = new double[hiddenDim];
            _mB1[c] = new double[hiddenDim];
            _vB1[c] = new double[hiddenDim];

            W2[c] = new double[hiddenDim];
            _mW2[c] = new double[hiddenDim];
            _vW2[c] = new double[hiddenDim];
            for (int j = 0; j < hiddenDim; j++)
            {
                W2[c][j] = (rng.NextDouble() * 2.0 - 1.0) * scale2;
            }

            B2[c] = 0.0;
        }
    }

    /// <summary>
    /// Computes predicted sigmoid probabilities for all classes independently.
    /// </summary>
    public double[] PredictProbabilities(ReadOnlySpan<double> features)
    {
        double[] probs = new double[ClassCount];
        for (int c = 0; c < ClassCount; c++)
        {
            double z = B2[c];
            var w2c = W2[c];
            var b1c = B1[c];
            var w1c = W1[c];

            for (int j = 0; j < HiddenDim; j++)
            {
                double sum = b1c[j];
                var w1cj = w1c[j];
                for (int f = 0; f < FeatureDim; f++)
                {
                    sum += w1cj[f] * features[f];
                }
                double hj = sum > 0.0 ? sum : 0.0; // ReLU
                z += w2c[j] * hj;
            }

            probs[c] = Sigmoid(z);
        }
        return probs;
    }

    /// <summary>
    /// Predicts the best class label and operation tally.
    /// </summary>
    public (int Prediction, int AccumulateOps) Predict(ReadOnlySpan<double> features, double[]? thresholds = null)
    {
        double[] probs = PredictProbabilities(features);
        int best = 0;
        double bestScore = double.NegativeInfinity;

        for (int c = 0; c < ClassCount; c++)
        {
            double t = (thresholds != null && c < thresholds.Length) ? Math.Max(1e-4, thresholds[c]) : 1.0;
            double score = probs[c] / t;
            if (score > bestScore)
            {
                bestScore = score;
                best = c;
            }
        }

        int ops = ClassCount * (HiddenDim * FeatureDim + HiddenDim);
        return (best, ops);
    }

    /// <summary>
    /// Trains each binary OvR head using Adam and positive class weighting over the shared feature vectors.
    /// </summary>
    public void Train(
        IReadOnlyList<(double[] Features, int Label)> dataset,
        int epochs = 25,
        double lr = 0.01,
        Action<int, int>? onProgress = null,
        double weightPower = 0.5)
    {
        if (dataset == null || dataset.Count == 0) return;

        int n = dataset.Count;
        double[] posWeights = new double[ClassCount];

        for (int c = 0; c < ClassCount; c++)
        {
            int pos = 0;
            for (int i = 0; i < n; i++)
            {
                if (dataset[i].Label == c) pos++;
            }
            int neg = n - pos;
            posWeights[c] = pos > 0 ? Math.Pow((double)neg / pos, weightPower) : 1.0;
        }

        for (int ep = 0; ep < epochs; ep++)
        {
            TrainEpoch(dataset, posWeights, lr);
            onProgress?.Invoke(ep + 1, epochs);
        }
    }

    /// <summary>
    /// Deep-clones the OvrTrainer including learned weights and all Adam optimizer first/second moments.
    /// </summary>
    public OvrTrainer Clone()
    {
        var clone = new OvrTrainer(ClassCount, FeatureDim, HiddenDim, Seed);
        for (int c = 0; c < ClassCount; c++)
        {
            for (int j = 0; j < HiddenDim; j++)
            {
                Array.Copy(W1[c][j], clone.W1[c][j], FeatureDim);
                Array.Copy(_mW1[c][j], clone._mW1[c][j], FeatureDim);
                Array.Copy(_vW1[c][j], clone._vW1[c][j], FeatureDim);
            }
            Array.Copy(B1[c], clone.B1[c], HiddenDim);
            Array.Copy(_mB1[c], clone._mB1[c], HiddenDim);
            Array.Copy(_vB1[c], clone._vB1[c], HiddenDim);

            Array.Copy(W2[c], clone.W2[c], HiddenDim);
            Array.Copy(_mW2[c], clone._mW2[c], HiddenDim);
            Array.Copy(_vW2[c], clone._vW2[c], HiddenDim);

            clone.B2[c] = B2[c];
            clone._mB2[c] = _mB2[c];
            clone._vB2[c] = _vB2[c];
            clone._adamStep[c] = _adamStep[c];
        }
        return clone;
    }

    /// <summary>
    /// Executes a single epoch of OvR training (useful for validation checkpointing).
    /// </summary>
    public void TrainEpoch(
        IReadOnlyList<(double[] Features, int Label)> dataset,
        double[] posWeights,
        double lr = 0.01)
    {
        int n = dataset.Count;
        const double beta1 = 0.9;
        const double beta2 = 0.999;
        const double eps = 1e-8;

        double[] h = new double[HiddenDim];
        double[] deltaH = new double[HiddenDim];

        for (int i = 0; i < n; i++)
        {
            var (feats, label) = dataset[i];

            for (int c = 0; c < ClassCount; c++)
            {
                double wPos = posWeights[c];
                double target = (label == c) ? 1.0 : 0.0;

                double z = B2[c];
                var w2c = W2[c];
                var b1c = B1[c];
                var w1c = W1[c];

                for (int j = 0; j < HiddenDim; j++)
                {
                    double sum = b1c[j];
                    var w1cj = w1c[j];
                    for (int f = 0; f < FeatureDim; f++)
                    {
                        sum += w1cj[f] * feats[f];
                    }
                    h[j] = sum > 0.0 ? sum : 0.0;
                    z += w2c[j] * h[j];
                }

                double p = Sigmoid(z);
                double gradZ = (target > 0.5) ? wPos * (p - 1.0) : p;

                for (int j = 0; j < HiddenDim; j++)
                {
                    deltaH[j] = (h[j] > 0.0) ? gradZ * w2c[j] : 0.0;
                }

                _adamStep[c]++;
                int step = _adamStep[c];
                double b1Corr = 1.0 - Math.Pow(beta1, step);
                double b2Corr = 1.0 - Math.Pow(beta2, step);
                double alpha = lr * Math.Sqrt(b2Corr) / Math.Max(1e-8, b1Corr);

                double mB2Val = beta1 * _mB2[c] + (1.0 - beta1) * gradZ;
                double vB2Val = beta2 * _vB2[c] + (1.0 - beta2) * gradZ * gradZ;
                _mB2[c] = mB2Val;
                _vB2[c] = vB2Val;
                B2[c] -= alpha * mB2Val / (Math.Sqrt(vB2Val) + eps);

                var mW2c = _mW2[c];
                var vW2c = _vW2[c];
                for (int j = 0; j < HiddenDim; j++)
                {
                    double gW2 = gradZ * h[j];
                    mW2c[j] = beta1 * mW2c[j] + (1.0 - beta1) * gW2;
                    vW2c[j] = beta2 * vW2c[j] + (1.0 - beta2) * gW2 * gW2;
                    w2c[j] -= alpha * mW2c[j] / (Math.Sqrt(vW2c[j]) + eps);
                }

                var mB1c = _mB1[c];
                var vB1c = _vB1[c];
                var mW1c = _mW1[c];
                var vW1c = _vW1[c];
                for (int j = 0; j < HiddenDim; j++)
                {
                    double dH = deltaH[j];
                    mB1c[j] = beta1 * mB1c[j] + (1.0 - beta1) * dH;
                    vB1c[j] = beta2 * vB1c[j] + (1.0 - beta2) * dH * dH;
                    b1c[j] -= alpha * mB1c[j] / (Math.Sqrt(vB1c[j]) + eps);

                    if (Math.Abs(dH) < 1e-12) continue;

                    var w1cj = w1c[j];
                    var mW1cj = mW1c[j];
                    var vW1cj = vW1c[j];
                    for (int f = 0; f < FeatureDim; f++)
                    {
                        double gW1 = dH * feats[f];
                        mW1cj[f] = beta1 * mW1cj[f] + (1.0 - beta1) * gW1;
                        vW1cj[f] = beta2 * vW1cj[f] + (1.0 - beta2) * gW1 * gW1;
                        w1cj[f] -= alpha * mW1cj[f] / (Math.Sqrt(vW1cj[f]) + eps);
                    }
                }
            }
        }
    }

    private static double Sigmoid(double z)
    {
        if (z < -15.0) return 0.0;
        if (z > 15.0) return 1.0;
        return 1.0 / (1.0 + Math.Exp(-z));
    }
}
