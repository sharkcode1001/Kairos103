namespace Kairos103.Core.Models;

/// <summary>
/// Supervised non-linear neuromorphic readout classifier with sparse backpropagation and Adam optimization.
/// Supports both 2-layer non-linear MLP readout (HiddenDim > 0) and direct linear margin readout (HiddenDim = 0).
/// Preserves exact temporal invariance because any downstream function f(X) inherits the invariance of X.
/// </summary>
public sealed class PairwiseTrainer
{
    public int ClassCount { get; }
    public int FeatureDim { get; }
    public int HiddenDim { get; }

    // Direct linear weights (for 1-layer compatibility or linear mode)
    public double[][] Weights { get; }
    public double[] Bias { get; }

    // 2-layer MLP weights (when HiddenDim > 0)
    public double[][] W1 { get; } // HiddenDim x FeatureDim
    public double[] B1 { get; }   // HiddenDim
    public double[][] W2 { get; } // ClassCount x HiddenDim
    public double[] B2 { get; }   // ClassCount

    // Adam optimizer moments
    private readonly double[][] _mW1, _vW1;
    private readonly double[] _mB1, _vB1;
    private readonly double[][] _mW2, _vW2;
    private readonly double[] _mB2, _vB2;
    private int _adamStep;

    public PairwiseTrainer(int classCount, int featureDim, int hiddenDim = 32, int seed = 42)
    {
        ClassCount = classCount;
        FeatureDim = featureDim;
        HiddenDim = hiddenDim;

        Weights = new double[classCount][];
        Bias = new double[classCount];
        for (int c = 0; c < classCount; c++)
        {
            Weights[c] = new double[featureDim];
        }

        if (hiddenDim > 0)
        {
            var rng = new Random(seed);
            double scale1 = Math.Sqrt(2.0 / featureDim);
            double scale2 = Math.Sqrt(2.0 / hiddenDim);

            W1 = new double[hiddenDim][];
            _mW1 = new double[hiddenDim][];
            _vW1 = new double[hiddenDim][];
            for (int j = 0; j < hiddenDim; j++)
            {
                W1[j] = new double[featureDim];
                _mW1[j] = new double[featureDim];
                _vW1[j] = new double[featureDim];
                for (int f = 0; f < featureDim; f++)
                {
                    W1[j][f] = (rng.NextDouble() * 2.0 - 1.0) * scale1;
                }
            }

            B1 = new double[hiddenDim];
            _mB1 = new double[hiddenDim];
            _vB1 = new double[hiddenDim];

            W2 = new double[classCount][];
            _mW2 = new double[classCount][];
            _vW2 = new double[classCount][];
            for (int c = 0; c < classCount; c++)
            {
                W2[c] = new double[hiddenDim];
                _mW2[c] = new double[hiddenDim];
                _vW2[c] = new double[hiddenDim];
                for (int j = 0; j < hiddenDim; j++)
                {
                    W2[c][j] = (rng.NextDouble() * 2.0 - 1.0) * scale2;
                }
            }

            B2 = new double[classCount];
            _mB2 = new double[classCount];
            _vB2 = new double[classCount];
        }
        else
        {
            W1 = []; _mW1 = []; _vW1 = [];
            B1 = []; _mB1 = []; _vB1 = [];
            W2 = []; _mW2 = []; _vW2 = [];
            B2 = []; _mB2 = []; _vB2 = [];
        }
    }

    /// <summary>
    /// Trains the classifier using Adam optimization with Cross-Entropy loss over sparse rank features.
    /// </summary>
    public void Train(IReadOnlyList<(double[] Features, int Label)> dataset, int epochs = 20, double lr = 0.01, Action<int, int>? onProgress = null)
    {
        if (dataset == null || dataset.Count == 0) return;

        if (HiddenDim > 0)
        {
            TrainMlp(dataset, epochs, lr, onProgress);
        }
        else
        {
            TrainLinear(dataset, epochs, lr, onProgress);
        }
    }

    private void TrainMlp(IReadOnlyList<(double[] Features, int Label)> dataset, int epochs, double lr, Action<int, int>? onProgress)
    {
        double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        double[] h = new double[HiddenDim];
        double[] logits = new double[ClassCount];
        double[] probs = new double[ClassCount];
        double[] grad2 = new double[ClassCount];
        double[] grad1 = new double[HiddenDim];

        List<int> activeFeatures = new List<int>(64);

        for (int ep = 0; ep < epochs; ep++)
        {
            onProgress?.Invoke(ep + 1, epochs);

            foreach (var (features, label) in dataset)
            {
                // Gather sparse active feature indices
                activeFeatures.Clear();
                for (int f = 0; f < FeatureDim; f++)
                {
                    if (features[f] > 0.0) activeFeatures.Add(f);
                }

                // Layer 1 forward: H = LeakyReLU(W1 * x + B1)
                for (int j = 0; j < HiddenDim; j++)
                {
                    double sum = B1[j];
                    for (int k = 0; k < activeFeatures.Count; k++)
                    {
                        int f = activeFeatures[k];
                        sum += W1[j][f] * features[f];
                    }
                    h[j] = sum > 0.0 ? sum : 0.01 * sum;
                }

                // Layer 2 forward: Logits = W2 * h + B2
                double maxLogit = double.NegativeInfinity;
                for (int c = 0; c < ClassCount; c++)
                {
                    double sum = B2[c];
                    for (int j = 0; j < HiddenDim; j++)
                    {
                        sum += W2[c][j] * h[j];
                    }
                    logits[c] = sum;
                    if (sum > maxLogit) maxLogit = sum;
                }

                // Softmax
                double sumExp = 0.0;
                for (int c = 0; c < ClassCount; c++)
                {
                    double exp = Math.Exp(logits[c] - maxLogit);
                    probs[c] = exp;
                    sumExp += exp;
                }
                double invSum = 1.0 / Math.Max(1e-12, sumExp);
                for (int c = 0; c < ClassCount; c++)
                {
                    probs[c] *= invSum;
                    grad2[c] = probs[c] - (c == label ? 1.0 : 0.0);
                }

                // Backprop to hidden layer
                for (int j = 0; j < HiddenDim; j++)
                {
                    double d = 0.0;
                    for (int c = 0; c < ClassCount; c++)
                    {
                        d += grad2[c] * W2[c][j];
                    }
                    grad1[j] = h[j] > 0.0 ? d : 0.01 * d;
                }

                // Adam update step
                _adamStep++;
                double b1Correction = 1.0 - Math.Pow(beta1, _adamStep);
                double b2Correction = 1.0 - Math.Pow(beta2, _adamStep);
                double alpha = lr * Math.Sqrt(b2Correction) / Math.Max(1e-8, b1Correction);

                // Update W2 and B2
                for (int c = 0; c < ClassCount; c++)
                {
                    double gB2 = grad2[c];
                    _mB2[c] = beta1 * _mB2[c] + (1.0 - beta1) * gB2;
                    _vB2[c] = beta2 * _vB2[c] + (1.0 - beta2) * gB2 * gB2;
                    B2[c] -= alpha * _mB2[c] / (Math.Sqrt(_vB2[c]) + eps);

                    for (int j = 0; j < HiddenDim; j++)
                    {
                        double gW2 = grad2[c] * h[j];
                        _mW2[c][j] = beta1 * _mW2[c][j] + (1.0 - beta1) * gW2;
                        _vW2[c][j] = beta2 * _vW2[c][j] + (1.0 - beta2) * gW2 * gW2;
                        W2[c][j] -= alpha * _mW2[c][j] / (Math.Sqrt(_vW2[c][j]) + eps);
                    }
                }

                // Update W1 and B1 (sparse update on active features only)
                for (int j = 0; j < HiddenDim; j++)
                {
                    double gB1 = grad1[j];
                    _mB1[j] = beta1 * _mB1[j] + (1.0 - beta1) * gB1;
                    _vB1[j] = beta2 * _vB1[j] + (1.0 - beta2) * gB1 * gB1;
                    B1[j] -= alpha * _mB1[j] / (Math.Sqrt(_vB1[j]) + eps);

                    for (int k = 0; k < activeFeatures.Count; k++)
                    {
                        int f = activeFeatures[k];
                        double gW1 = grad1[j] * features[f];
                        _mW1[j][f] = beta1 * _mW1[j][f] + (1.0 - beta1) * gW1;
                        _vW1[j][f] = beta2 * _vW1[j][f] + (1.0 - beta2) * gW1 * gW1;
                        W1[j][f] -= alpha * _mW1[j][f] / (Math.Sqrt(_vW1[j][f]) + eps);
                    }
                }
            }
        }
    }

    private void TrainLinear(IReadOnlyList<(double[] Features, int Label)> dataset, int epochs, double lr, Action<int, int>? onProgress)
    {
        double[] scores = new double[ClassCount];

        for (int ep = 0; ep < epochs; ep++)
        {
            onProgress?.Invoke(ep + 1, epochs);
            foreach (var (features, label) in dataset)
            {
                for (int c = 0; c < ClassCount; c++)
                {
                    double s = Bias[c];
                    for (int f = 0; f < FeatureDim; f++)
                    {
                        if (features[f] > 0.0) s += Weights[c][f] * features[f];
                    }
                    scores[c] = s;
                }

                int predicted = 0;
                double maxScore = scores[0];
                for (int c = 1; c < ClassCount; c++)
                {
                    if (scores[c] > maxScore)
                    {
                        maxScore = scores[c];
                        predicted = c;
                    }
                }

                if (predicted != label)
                {
                    Bias[label] += lr;
                    Bias[predicted] -= lr;
                    for (int f = 0; f < FeatureDim; f++)
                    {
                        if (features[f] > 0.0)
                        {
                            Weights[label][f] += lr * features[f];
                            Weights[predicted][f] -= lr * features[f];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Infers the predicted class for a feature vector and tallies required operations.
    /// </summary>
    public (int Prediction, int AccumulateOps) Predict(double[] features)
    {
        int accOps = 0;

        if (HiddenDim > 0)
        {
            // Layer 1: H = LeakyReLU(W1 * x + B1)
            double[] h = new double[HiddenDim];
            for (int j = 0; j < HiddenDim; j++)
            {
                double sum = B1[j];
                for (int f = 0; f < FeatureDim; f++)
                {
                    if (features[f] > 0.0)
                    {
                        sum += W1[j][f] * features[f];
                        accOps++;
                    }
                }
                h[j] = sum > 0.0 ? sum : 0.01 * sum;
            }

            // Layer 2: Logits = W2 * h + B2
            int bestClass = 0;
            double bestLogit = double.NegativeInfinity;
            for (int c = 0; c < ClassCount; c++)
            {
                double sum = B2[c];
                for (int j = 0; j < HiddenDim; j++)
                {
                    sum += W2[c][j] * h[j];
                    accOps++;
                }
                if (sum > bestLogit)
                {
                    bestLogit = sum;
                    bestClass = c;
                }
            }

            return (bestClass, accOps);
        }
        else
        {
            int bestClass = 0;
            double bestScore = double.NegativeInfinity;

            for (int c = 0; c < ClassCount; c++)
            {
                double score = Bias[c];
                for (int f = 0; f < FeatureDim; f++)
                {
                    if (features[f] > 0.0)
                    {
                        score += Weights[c][f] * features[f];
                        accOps++;
                    }
                }

                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            return (bestClass, accOps);
        }
    }
}
