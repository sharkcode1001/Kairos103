using Kairos103.Core.Data;

namespace Kairos103.Core.Baselines;

/// <summary>
/// Common contract for native ECG baseline classifiers.
/// </summary>
public interface IEcgBaseline
{
    string Name { get; }
    void Fit(IReadOnlyList<KaggleEcgSample> samples);
    int Predict(KaggleEcgSample sample);
}

/// <summary>
/// Baseline 1: Majority Class Dummy Classifier.
/// Always predicts the most frequent class from the training set.
/// Demonstrates the metric floor under extreme class imbalance.
/// </summary>
public sealed class MajorityDummyClassifier : IEcgBaseline
{
    public string Name => "Majority Dummy Classifier";
    private int _majorityClass;

    public void Fit(IReadOnlyList<KaggleEcgSample> samples)
    {
        if (samples == null || samples.Count == 0) return;
        _majorityClass = samples
            .GroupBy(s => s.Label)
            .OrderByDescending(g => g.Count())
            .First()
            .Key;
    }

    public int Predict(KaggleEcgSample sample) => _majorityClass;
}

/// <summary>
/// Baseline 2: Multiclass Linear Logistic Regression with Adam optimizer.
/// Operates directly on standardized raw waveform points and rhythm features.
/// </summary>
public sealed class LinearEcgClassifier : IEcgBaseline
{
    public string Name => "Standardized Linear Logistic Classifier";

    private const int FeatureDim = KaggleEcgSample.TotalLength; // 253
    private const int ClassCount = 4;

    private readonly double[][] _w = new double[ClassCount][];
    private readonly double[] _b = new double[ClassCount];
    private readonly double[] _mean = new double[FeatureDim];
    private readonly double[] _std = new double[FeatureDim];

    public LinearEcgClassifier()
    {
        for (int c = 0; c < ClassCount; c++)
        {
            _w[c] = new double[FeatureDim];
        }
    }

    public void Fit(IReadOnlyList<KaggleEcgSample> samples)
    {
        if (samples == null || samples.Count == 0) return;

        // 1. Compute normalization statistics
        int n = samples.Count;
        for (int f = 0; f < FeatureDim; f++)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                sum += GetFeature(samples[i], f);
            }
            _mean[f] = sum / n;

            double varSum = 0.0;
            for (int i = 0; i < n; i++)
            {
                double diff = GetFeature(samples[i], f) - _mean[f];
                varSum += diff * diff;
            }
            _std[f] = Math.Sqrt(varSum / n) + 1e-6;
        }

        // 2. Class weights for balanced gradients
        var classWeights = KaggleEcgDataset.ComputeBalancedClassWeights(samples, ClassCount);

        // 3. Adam optimization for 15 epochs
        int epochs = 15;
        double lr = 0.005;
        double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        var mW = new double[ClassCount][];
        var vW = new double[ClassCount][];
        var mB = new double[ClassCount];
        var vB = new double[ClassCount];
        for (int c = 0; c < ClassCount; c++)
        {
            mW[c] = new double[FeatureDim];
            vW[c] = new double[FeatureDim];
        }

        int step = 0;
        double[] x = new double[FeatureDim];
        double[] logits = new double[ClassCount];
        double[] probs = new double[ClassCount];
        double[] grad = new double[ClassCount];

        for (int ep = 0; ep < epochs; ep++)
        {
            for (int i = 0; i < n; i++)
            {
                var s = samples[i];
                for (int f = 0; f < FeatureDim; f++)
                {
                    x[f] = (GetFeature(s, f) - _mean[f]) / _std[f];
                }

                double maxLogit = double.NegativeInfinity;
                for (int c = 0; c < ClassCount; c++)
                {
                    double sum = _b[c];
                    for (int f = 0; f < FeatureDim; f++)
                    {
                        sum += _w[c][f] * x[f];
                    }
                    logits[c] = sum;
                    if (sum > maxLogit) maxLogit = sum;
                }

                double sumExp = 0.0;
                for (int c = 0; c < ClassCount; c++)
                {
                    double e = Math.Exp(logits[c] - maxLogit);
                    probs[c] = e;
                    sumExp += e;
                }

                double invSum = 1.0 / Math.Max(1e-12, sumExp);
                double weight = (s.Label >= 0 && s.Label < ClassCount) ? classWeights[s.Label] : 1.0;
                for (int c = 0; c < ClassCount; c++)
                {
                    probs[c] *= invSum;
                    double target = (c == s.Label) ? 1.0 : 0.0;
                    grad[c] = weight * (probs[c] - target);
                }

                step++;
                double b1Corr = 1.0 - Math.Pow(beta1, step);
                double b2Corr = 1.0 - Math.Pow(beta2, step);
                double alpha = lr * Math.Sqrt(b2Corr) / Math.Max(1e-8, b1Corr);

                for (int c = 0; c < ClassCount; c++)
                {
                    double gB = grad[c];
                    mB[c] = beta1 * mB[c] + (1.0 - beta1) * gB;
                    vB[c] = beta2 * vB[c] + (1.0 - beta2) * gB * gB;
                    _b[c] -= alpha * mB[c] / (Math.Sqrt(vB[c]) + eps);

                    for (int f = 0; f < FeatureDim; f++)
                    {
                        double gW = grad[c] * x[f];
                        mW[c][f] = beta1 * mW[c][f] + (1.0 - beta1) * gW;
                        vW[c][f] = beta2 * vW[c][f] + (1.0 - beta2) * gW * gW;
                        _w[c][f] -= alpha * mW[c][f] / (Math.Sqrt(vW[c][f]) + eps);
                    }
                }
            }
        }
    }

    public int Predict(KaggleEcgSample sample)
    {
        int best = 0;
        double bestLogit = double.NegativeInfinity;

        for (int c = 0; c < ClassCount; c++)
        {
            double sum = _b[c];
            for (int f = 0; f < FeatureDim; f++)
            {
                double x = (GetFeature(sample, f) - _mean[f]) / _std[f];
                sum += _w[c][f] * x;
            }
            if (sum > bestLogit)
            {
                bestLogit = sum;
                best = c;
            }
        }

        return best;
    }

    private static double GetFeature(KaggleEcgSample s, int index)
    {
        if (index < KaggleEcgSample.WaveformLength)
            return s.Waveform[index];
        if (index == KaggleEcgSample.WaveformLength)
            return s.PreRr;
        if (index == KaggleEcgSample.WaveformLength + 1)
            return s.PostRr;
        return s.RrRatio;
    }
}

/// <summary>
/// Baseline 3: Native 1D Convolutional Neural Network.
/// Convolves 8 learned kernels over the temporal ECG trace, applies ReLU + MaxPool,
/// and feeds the pooled representation with rhythm features into a classification layer.
/// </summary>
public sealed class Cnn1DEcgClassifier : IEcgBaseline
{
    public string Name => "Native 1D CNN Classifier (Conv1D + MaxPool + FC)";

    private const int InLength = KaggleEcgSample.WaveformLength; // 250
    private const int Filters = 8;
    private const int KernelSize = 7;
    private const int Stride = 2;
    // Conv output length: (250 - 7) / 2 + 1 = 122
    private const int ConvOutLen = 122;
    // MaxPool 2x2: 122 / 2 = 61
    private const int PoolOutLen = 61;
    private const int FlatDim = Filters * PoolOutLen; // 8 * 61 = 488
    private const int TotalFcIn = FlatDim + KaggleEcgSample.AuxCount; // 491
    private const int ClassCount = 4;

    private readonly double[][] _kernels = new double[Filters][];
    private readonly double[] _convBiases = new double[Filters];
    private readonly double[][] _fcWeights = new double[ClassCount][];
    private readonly double[] _fcBiases = new double[ClassCount];

    public Cnn1DEcgClassifier(int seed = 42)
    {
        var rng = new Random(seed);
        double kScale = Math.Sqrt(2.0 / KernelSize);
        for (int k = 0; k < Filters; k++)
        {
            _kernels[k] = new double[KernelSize];
            for (int i = 0; i < KernelSize; i++)
            {
                _kernels[k][i] = (rng.NextDouble() * 2.0 - 1.0) * kScale;
            }
        }

        double fcScale = Math.Sqrt(2.0 / TotalFcIn);
        for (int c = 0; c < ClassCount; c++)
        {
            _fcWeights[c] = new double[TotalFcIn];
            for (int f = 0; f < TotalFcIn; f++)
            {
                _fcWeights[c][f] = (rng.NextDouble() * 2.0 - 1.0) * fcScale;
            }
        }
    }

    public void Fit(IReadOnlyList<KaggleEcgSample> samples)
    {
        if (samples == null || samples.Count == 0) return;

        var classWeights = KaggleEcgDataset.ComputeBalancedClassWeights(samples, ClassCount);
        int epochs = 10;
        double lr = 0.003;
        double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

        var mFcW = new double[ClassCount][];
        var vFcW = new double[ClassCount][];
        var mFcB = new double[ClassCount];
        var vFcB = new double[ClassCount];
        for (int c = 0; c < ClassCount; c++)
        {
            mFcW[c] = new double[TotalFcIn];
            vFcW[c] = new double[TotalFcIn];
        }

        int step = 0;
        double[] convOut = new double[Filters * ConvOutLen];
        double[] poolOut = new double[TotalFcIn];
        double[] logits = new double[ClassCount];
        double[] probs = new double[ClassCount];
        double[] grad = new double[ClassCount];

        for (int ep = 0; ep < epochs; ep++)
        {
            for (int i = 0; i < samples.Count; i++)
            {
                var s = samples[i];
                ForwardFeatures(s, poolOut);

                // FC forward
                double maxLogit = double.NegativeInfinity;
                for (int c = 0; c < ClassCount; c++)
                {
                    double sum = _fcBiases[c];
                    for (int f = 0; f < TotalFcIn; f++)
                    {
                        sum += _fcWeights[c][f] * poolOut[f];
                    }
                    logits[c] = sum;
                    if (sum > maxLogit) maxLogit = sum;
                }

                // Softmax
                double sumExp = 0.0;
                for (int c = 0; c < ClassCount; c++)
                {
                    double e = Math.Exp(logits[c] - maxLogit);
                    probs[c] = e;
                    sumExp += e;
                }
                double invSum = 1.0 / Math.Max(1e-12, sumExp);
                double weight = (s.Label >= 0 && s.Label < ClassCount) ? classWeights[s.Label] : 1.0;
                for (int c = 0; c < ClassCount; c++)
                {
                    probs[c] *= invSum;
                    double target = (c == s.Label) ? 1.0 : 0.0;
                    grad[c] = weight * (probs[c] - target);
                }

                step++;
                double b1Corr = 1.0 - Math.Pow(beta1, step);
                double b2Corr = 1.0 - Math.Pow(beta2, step);
                double alpha = lr * Math.Sqrt(b2Corr) / Math.Max(1e-8, b1Corr);

                for (int c = 0; c < ClassCount; c++)
                {
                    double gB = grad[c];
                    mFcB[c] = beta1 * mFcB[c] + (1.0 - beta1) * gB;
                    vFcB[c] = beta2 * vFcB[c] + (1.0 - beta2) * gB * gB;
                    _fcBiases[c] -= alpha * mFcB[c] / (Math.Sqrt(vFcB[c]) + eps);

                    for (int f = 0; f < TotalFcIn; f++)
                    {
                        double gW = grad[c] * poolOut[f];
                        mFcW[c][f] = beta1 * mFcW[c][f] + (1.0 - beta1) * gW;
                        vFcW[c][f] = beta2 * vFcW[c][f] + (1.0 - beta2) * gW * gW;
                        _fcWeights[c][f] -= alpha * mFcW[c][f] / (Math.Sqrt(vFcW[c][f]) + eps);
                    }
                }
            }
        }
    }

    public int Predict(KaggleEcgSample sample)
    {
        double[] poolOut = new double[TotalFcIn];
        ForwardFeatures(sample, poolOut);

        int best = 0;
        double bestLogit = double.NegativeInfinity;
        for (int c = 0; c < ClassCount; c++)
        {
            double sum = _fcBiases[c];
            for (int f = 0; f < TotalFcIn; f++)
            {
                sum += _fcWeights[c][f] * poolOut[f];
            }
            if (sum > bestLogit)
            {
                bestLogit = sum;
                best = c;
            }
        }

        return best;
    }

    private void ForwardFeatures(KaggleEcgSample s, double[] outputFcIn)
    {
        // 1. Conv1D + ReLU + MaxPool
        int outIdx = 0;
        for (int k = 0; k < Filters; k++)
        {
            var kernel = _kernels[k];
            double bias = _convBiases[k];

            for (int p = 0; p < PoolOutLen; p++)
            {
                int convPos1 = p * 2;
                int convPos2 = p * 2 + 1;

                int waveIdx1 = convPos1 * Stride;
                double sum1 = bias;
                for (int ki = 0; ki < KernelSize; ki++)
                {
                    int wi = waveIdx1 + ki;
                    if (wi < InLength) sum1 += kernel[ki] * s.Waveform[wi];
                }
                double act1 = sum1 > 0.0 ? sum1 : 0.0;

                int waveIdx2 = convPos2 * Stride;
                double sum2 = bias;
                for (int ki = 0; ki < KernelSize; ki++)
                {
                    int wi = waveIdx2 + ki;
                    if (wi < InLength) sum2 += kernel[ki] * s.Waveform[wi];
                }
                double act2 = sum2 > 0.0 ? sum2 : 0.0;

                outputFcIn[outIdx++] = Math.Max(act1, act2);
            }
        }

        // 2. Append auxiliary features
        outputFcIn[FlatDim] = s.PreRr;
        outputFcIn[FlatDim + 1] = s.PostRr;
        outputFcIn[FlatDim + 2] = s.RrRatio;
    }
}
