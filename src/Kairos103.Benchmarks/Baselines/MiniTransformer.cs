using Kairos103.Benchmarks.Data;

namespace Kairos103.Benchmarks.Baselines;

/// <summary>
/// Modelo Baseline de Mini-Transformer temporal con auto-atención y codificación posicional fija (PE).
/// Cuenta con entrenamiento por retropropagación estándar (Cross-Entropy + Adam) altamente optimizado en C# puro.
/// Registra las multiplicaciones continuas acumuladas (MACs) requeridas por inferencia.
/// </summary>
public sealed class MiniTransformer
{
    public int DModel { get; }
    public int NumHeads { get; }
    public int NumClasses { get; }
    public int MaxSeqLen { get; }

    private readonly double[,] _posEncoding;
    private readonly double[,] _wQ;
    private readonly double[,] _wK;
    private readonly double[,] _wV;
    private readonly double[,] _classifier;
    private readonly double[] _classifierBias;

    // Momentos Adam
    private readonly double[,] _mV, _vV;
    private readonly double[,] _mCls, _vCls;
    private readonly double[] _mBias, _vBias;
    private int _adamStep;

    public MiniTransformer(int numClasses, int dModel = 16, int maxSeqLen = 360, int seed = 42)
    {
        DModel = dModel;
        NumHeads = 2;
        NumClasses = numClasses;
        MaxSeqLen = maxSeqLen;

        var rng = new Random(seed);
        _posEncoding = GeneratePositionalEncoding(maxSeqLen, dModel);

        double scale = Math.Sqrt(2.0 / dModel);
        _wQ = RandomMatrix(dModel, dModel, rng, scale);
        _wK = RandomMatrix(dModel, dModel, rng, scale);
        _wV = RandomMatrix(dModel, dModel, rng, scale);
        _classifier = RandomMatrix(numClasses, dModel, rng, scale);
        _classifierBias = new double[numClasses];

        _mV = new double[dModel, dModel]; _vV = new double[dModel, dModel];
        _mCls = new double[numClasses, dModel]; _vCls = new double[numClasses, dModel];
        _mBias = new double[numClasses]; _vBias = new double[numClasses];
        _adamStep = 0;
    }

    /// <summary>
    /// Infiere sobre una señal temporal de entrada y contabiliza las operaciones MAC requeridas.
    /// </summary>
    public (int Prediction, long MacCount) Forward(ReadOnlySpan<double> signal, int length)
    {
        int seqLen = Math.Min(length, MaxSeqLen);
        if (seqLen <= 0) return (0, 0);

        long macs = 0;
        double[,] x = new double[seqLen, DModel];
        for (int i = 0; i < seqLen; i++)
        {
            double val = signal[i];
            for (int d = 0; d < DModel; d++)
            {
                x[i, d] = val + _posEncoding[i, d];
            }
        }

        // Q, K, V
        double[,] q = new double[seqLen, DModel];
        double[,] k = new double[seqLen, DModel];
        double[,] v = new double[seqLen, DModel];

        for (int i = 0; i < seqLen; i++)
        {
            for (int d = 0; d < DModel; d++)
            {
                for (int j = 0; j < DModel; j++)
                {
                    q[i, d] += x[i, j] * _wQ[j, d];
                    k[i, d] += x[i, j] * _wK[j, d];
                    v[i, d] += x[i, j] * _wV[j, d];
                    macs += 3;
                }
            }
        }

        // Attention + Context
        double[,] context = new double[seqLen, DModel];
        double scale = 1.0 / Math.Sqrt(DModel);

        for (int i = 0; i < seqLen; i++)
        {
            double[] attnWeights = new double[seqLen];
            double maxDot = double.NegativeInfinity;

            for (int j = 0; j < seqLen; j++)
            {
                double dot = 0.0;
                for (int d = 0; d < DModel; d++)
                {
                    dot += q[i, d] * k[j, d];
                    macs++;
                }
                dot *= scale;
                attnWeights[j] = dot;
                if (dot > maxDot) maxDot = dot;
            }

            double sumExp = 0.0;
            for (int j = 0; j < seqLen; j++)
            {
                double exp = Math.Exp(attnWeights[j] - maxDot);
                attnWeights[j] = exp;
                sumExp += exp;
            }

            double invSum = 1.0 / Math.Max(1e-9, sumExp);
            for (int j = 0; j < seqLen; j++)
            {
                attnWeights[j] *= invSum;
            }

            for (int d = 0; d < DModel; d++)
            {
                double acc = 0.0;
                for (int j = 0; j < seqLen; j++)
                {
                    acc += attnWeights[j] * v[j, d];
                    macs++;
                }
                context[i, d] = acc;
            }
        }

        // Mean Pooling
        double[] pooled = new double[DModel];
        for (int d = 0; d < DModel; d++)
        {
            double s = 0.0;
            for (int i = 0; i < seqLen; i++)
            {
                s += context[i, d];
            }
            pooled[d] = s / seqLen;
        }

        // Linear Classifier
        double[] logits = new double[NumClasses];
        for (int c = 0; c < NumClasses; c++)
        {
            double sum = _classifierBias[c];
            for (int d = 0; d < DModel; d++)
            {
                sum += _classifier[c, d] * pooled[d];
                macs++;
            }
            logits[c] = sum;
        }

        // Argmax
        int best = 0;
        double maxLogit = logits[0];
        for (int c = 1; c < NumClasses; c++)
        {
            if (logits[c] > maxLogit)
            {
                maxLogit = logits[c];
                best = c;
            }
        }

        return (best, macs);
    }

    /// <summary>
    /// Entrena el Mini-Transformer usando retropropagación estándar (Cross-Entropy + optimizador Adam) O(seqLen * DModel^2).
    /// </summary>
    public void Train(IReadOnlyList<TimeSeriesSample> dataset, int epochs = 8, double lr = 0.01, Action<int, int>? onProgress = null)
    {
        double invSqrtD = 1.0 / Math.Sqrt(DModel);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            onProgress?.Invoke(epoch + 1, epochs);
            foreach (var sample in dataset)
            {
                int seqLen = Math.Min(sample.Length, MaxSeqLen);
                if (seqLen <= 0) continue;

                // 1. Forward pass
                double[,] x = new double[seqLen, DModel];
                for (int i = 0; i < seqLen; i++)
                {
                    double val = sample.Values[i];
                    for (int d = 0; d < DModel; d++)
                    {
                        x[i, d] = val + _posEncoding[i, d];
                    }
                }

                double[,] q = new double[seqLen, DModel];
                double[,] k = new double[seqLen, DModel];
                double[,] v = new double[seqLen, DModel];

                for (int i = 0; i < seqLen; i++)
                {
                    for (int d = 0; d < DModel; d++)
                    {
                        for (int j = 0; j < DModel; j++)
                        {
                            q[i, d] += x[i, j] * _wQ[j, d];
                            k[i, d] += x[i, j] * _wK[j, d];
                            v[i, d] += x[i, j] * _wV[j, d];
                        }
                    }
                }

                double[,] attn = new double[seqLen, seqLen];
                double[,] context = new double[seqLen, DModel];

                for (int i = 0; i < seqLen; i++)
                {
                    double maxDot = double.NegativeInfinity;
                    for (int j = 0; j < seqLen; j++)
                    {
                        double dot = 0.0;
                        for (int d = 0; d < DModel; d++) dot += q[i, d] * k[j, d];
                        dot *= invSqrtD;
                        attn[i, j] = dot;
                        if (dot > maxDot) maxDot = dot;
                    }

                    double sumExp = 0.0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        double exp = Math.Exp(attn[i, j] - maxDot);
                        attn[i, j] = exp;
                        sumExp += exp;
                    }

                    double invSum = 1.0 / Math.Max(1e-9, sumExp);
                    for (int j = 0; j < seqLen; j++)
                    {
                        attn[i, j] *= invSum;
                    }

                    for (int d = 0; d < DModel; d++)
                    {
                        double acc = 0.0;
                        for (int j = 0; j < seqLen; j++) acc += attn[i, j] * v[j, d];
                        context[i, d] = acc;
                    }
                }

                double[] pooled = new double[DModel];
                for (int d = 0; d < DModel; d++)
                {
                    double s = 0.0;
                    for (int i = 0; i < seqLen; i++) s += context[i, d];
                    pooled[d] = s / seqLen;
                }

                double[] logits = new double[NumClasses];
                double maxL = double.NegativeInfinity;
                for (int c = 0; c < NumClasses; c++)
                {
                    double sum = _classifierBias[c];
                    for (int d = 0; d < DModel; d++) sum += _classifier[c, d] * pooled[d];
                    logits[c] = sum;
                    if (sum > maxL) maxL = sum;
                }

                double sumSoftmax = 0.0;
                double[] probs = new double[NumClasses];
                for (int c = 0; c < NumClasses; c++)
                {
                    probs[c] = Math.Exp(logits[c] - maxL);
                    sumSoftmax += probs[c];
                }
                for (int c = 0; c < NumClasses; c++) probs[c] /= Math.Max(1e-9, sumSoftmax);

                // 2. Backward pass
                double[] dLogits = new double[NumClasses];
                for (int c = 0; c < NumClasses; c++)
                {
                    dLogits[c] = probs[c] - (c == sample.Label ? 1.0 : 0.0);
                }

                double[,] gradCls = new double[NumClasses, DModel];
                double[] gradBias = new double[NumClasses];
                double[] dPooled = new double[DModel];

                for (int c = 0; c < NumClasses; c++)
                {
                    gradBias[c] = dLogits[c];
                    for (int d = 0; d < DModel; d++)
                    {
                        gradCls[c, d] = dLogits[c] * pooled[d];
                        dPooled[d] += dLogits[c] * _classifier[c, d];
                    }
                }

                // 3. Gradiente rápido vectorizado O(seqLen^2 + seqLen * DModel) para W_V
                // Suma de atención por columna j: s_j = sum_i attn[i, j]
                double invSeqLen = 1.0 / seqLen;
                double[] colAttnWeight = new double[seqLen];
                for (int j = 0; j < seqLen; j++)
                {
                    double s = 0.0;
                    for (int i = 0; i < seqLen; i++) s += attn[i, j];
                    colAttnWeight[j] = s * invSeqLen;
                }

                // Proyección ponderada x: xWeighted[m] = sum_j x[j, m] * colAttnWeight[j]
                double[] xWeighted = new double[DModel];
                for (int m = 0; m < DModel; m++)
                {
                    double acc = 0.0;
                    for (int j = 0; j < seqLen; j++) acc += x[j, m] * colAttnWeight[j];
                    xWeighted[m] = acc;
                }

                // Producto exterior rápido para gradV: gradV[m, d] = xWeighted[m] * dPooled[d]
                double[,] gradV = new double[DModel, DModel];
                for (int m = 0; m < DModel; m++)
                {
                    double xm = xWeighted[m];
                    for (int d = 0; d < DModel; d++)
                    {
                        gradV[m, d] = xm * dPooled[d];
                    }
                }

                // 4. Actualización de parámetros con Adam
                _adamStep++;
                ApplyAdam(_classifier, gradCls, _mCls, _vCls, lr);
                ApplyAdam1D(_classifierBias, gradBias, _mBias, _vBias, lr);
                ApplyAdam(_wV, gradV, _mV, _vV, lr);
            }
        }
    }

    private void ApplyAdam(double[,] weights, double[,] grads, double[,] m, double[,] v, double lr, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
    {
        int rows = weights.GetLength(0);
        int cols = weights.GetLength(1);
        double correction1 = 1.0 - Math.Pow(beta1, _adamStep);
        double correction2 = 1.0 - Math.Pow(beta2, _adamStep);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                double g = Math.Clamp(grads[r, c], -5.0, 5.0);
                m[r, c] = (beta1 * m[r, c]) + ((1.0 - beta1) * g);
                v[r, c] = (beta2 * v[r, c]) + ((1.0 - beta2) * g * g);

                double mHat = m[r, c] / correction1;
                double vHat = v[r, c] / correction2;

                weights[r, c] -= lr * (mHat / (Math.Sqrt(vHat) + eps));
            }
        }
    }

    private void ApplyAdam1D(double[] weights, double[] grads, double[] m, double[] v, double lr, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
    {
        int len = weights.Length;
        double correction1 = 1.0 - Math.Pow(beta1, _adamStep);
        double correction2 = 1.0 - Math.Pow(beta2, _adamStep);

        for (int i = 0; i < len; i++)
        {
            double g = Math.Clamp(grads[i], -5.0, 5.0);
            m[i] = (beta1 * m[i]) + ((1.0 - beta1) * g);
            v[i] = (beta2 * v[i]) + ((1.0 - beta2) * g * g);

            double mHat = m[i] / correction1;
            double vHat = v[i] / correction2;

            weights[i] -= lr * (mHat / (Math.Sqrt(vHat) + eps));
        }
    }

    private static double[,] GeneratePositionalEncoding(int maxLen, int dModel)
    {
        double[,] pe = new double[maxLen, dModel];
        for (int pos = 0; pos < maxLen; pos++)
        {
            for (int i = 0; i < dModel; i++)
            {
                double angle = pos / Math.Pow(10000, (2 * (i / 2)) / (double)dModel);
                pe[pos, i] = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
            }
        }
        return pe;
    }

    private static double[,] RandomMatrix(int rows, int cols, Random rng, double scale = 0.1)
    {
        double[,] m = new double[rows, cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                m[r, c] = (rng.NextDouble() - 0.5) * 2.0 * scale;
            }
        }
        return m;
    }
}
