using Kairos103.Benchmarks.Data;

namespace Kairos103.Benchmarks.Baselines;

/// <summary>
/// Baseline de red recurrente LSTM clásica (Long Short-Term Memory).
/// Cuenta con entrenamiento por retropropagación a través del tiempo (BPTT + Adam) en C# puro sin librerías externas.
/// Procesa secuencias paso a paso actualizando celdas de memoria y compuertas (i, f, o, g).
/// </summary>
public sealed class SimpleLstm
{
    public int HiddenDim { get; }
    public int NumClasses { get; }

    private readonly double[,] _wX; // Entrada -> Compuertas (4 * HiddenDim x 1)
    private readonly double[,] _wH; // Oculto -> Compuertas (4 * HiddenDim x HiddenDim)
    private readonly double[] _b;
    private readonly double[,] _classifier;
    private readonly double[] _classifierBias;

    // Momentos Adam
    private readonly double[,] _mX, _vX;
    private readonly double[,] _mH, _vH;
    private readonly double[] _mB, _vB;
    private readonly double[,] _mCls, _vCls;
    private readonly double[] _mBias, _vBias;
    private int _adamStep;

    public SimpleLstm(int numClasses, int hiddenDim = 16, int seed = 42)
    {
        HiddenDim = hiddenDim;
        NumClasses = numClasses;
        int gatesDim = 4 * hiddenDim;

        var rng = new Random(seed);
        double scale = Math.Sqrt(2.0 / hiddenDim);
        _wX = RandomMatrix(gatesDim, 1, rng, scale);
        _wH = RandomMatrix(gatesDim, hiddenDim, rng, scale);
        _b = new double[gatesDim];
        _classifier = RandomMatrix(numClasses, hiddenDim, rng, scale);
        _classifierBias = new double[numClasses];

        _mX = new double[gatesDim, 1]; _vX = new double[gatesDim, 1];
        _mH = new double[gatesDim, hiddenDim]; _vH = new double[gatesDim, hiddenDim];
        _mB = new double[gatesDim]; _vB = new double[gatesDim];
        _mCls = new double[numClasses, hiddenDim]; _vCls = new double[numClasses, hiddenDim];
        _mBias = new double[numClasses]; _vBias = new double[numClasses];
        _adamStep = 0;
    }

    /// <summary>
    /// Infiere sobre una serie temporal paso a paso y contabiliza las operaciones MAC requeridas.
    /// </summary>
    public (int Prediction, long MacCount) Forward(ReadOnlySpan<double> signal, int length)
    {
        int seqLen = Math.Min(signal.Length, length);
        if (seqLen <= 0) return (0, 0);

        long macs = 0;
        double[] h = new double[HiddenDim];
        double[] c = new double[HiddenDim];
        int gatesDim = 4 * HiddenDim;

        for (int t = 0; t < seqLen; t++)
        {
            double x = signal[t];
            double[] gates = new double[gatesDim];

            // W_x * x + W_h * h + b
            for (int g = 0; g < gatesDim; g++)
            {
                gates[g] = (_wX[g, 0] * x) + _b[g];
                macs++;

                for (int j = 0; j < HiddenDim; j++)
                {
                    gates[g] += _wH[g, j] * h[j];
                    macs++;
                }
            }

            // Compuertas: i, f, o, g
            for (int j = 0; j < HiddenDim; j++)
            {
                double iGate = Sigmoid(gates[j]);
                double fGate = Sigmoid(gates[HiddenDim + j]);
                double oGate = Sigmoid(gates[(2 * HiddenDim) + j]);
                double gCandidate = Math.Tanh(gates[(3 * HiddenDim) + j]);

                c[j] = (fGate * c[j]) + (iGate * gCandidate);
                h[j] = oGate * Math.Tanh(c[j]);
                macs += 3;
            }
        }

        // Clasificación final sobre último estado h
        double[] logits = new double[NumClasses];
        for (int k = 0; k < NumClasses; k++)
        {
            double sum = _classifierBias[k];
            for (int j = 0; j < HiddenDim; j++)
            {
                sum += _classifier[k, j] * h[j];
                macs++;
            }
            logits[k] = sum;
        }

        int best = 0;
        double maxVal = logits[0];
        for (int k = 1; k < NumClasses; k++)
        {
            if (logits[k] > maxVal)
            {
                maxVal = logits[k];
                best = k;
            }
        }

        return (best, macs);
    }

    /// <summary>
    /// Entrena la red LSTM usando BPTT (Backpropagation Through Time) con optimizador Adam en C# puro.
    /// </summary>
    public void Train(IReadOnlyList<TimeSeriesSample> dataset, int epochs = 15, double lr = 0.005)
    {
        int gatesDim = 4 * HiddenDim;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var sample in dataset)
            {
                int seqLen = Math.Min(sample.Length, sample.Values.Length);
                if (seqLen <= 0) continue;

                // 1. Forward pass almacenando trayectorias temporales
                double[][] hHist = new double[seqLen + 1][];
                double[][] cHist = new double[seqLen + 1][];
                hHist[0] = new double[HiddenDim];
                cHist[0] = new double[HiddenDim];

                double[][] iGates = new double[seqLen][];
                double[][] fGates = new double[seqLen][];
                double[][] oGates = new double[seqLen][];
                double[][] gCands = new double[seqLen][];

                for (int t = 0; t < seqLen; t++)
                {
                    double x = sample.Values[t];
                    double[] hPrev = hHist[t];
                    double[] cPrev = cHist[t];

                    double[] hNext = new double[HiddenDim];
                    double[] cNext = new double[HiddenDim];

                    double[] gates = new double[gatesDim];
                    for (int g = 0; g < gatesDim; g++)
                    {
                        gates[g] = (_wX[g, 0] * x) + _b[g];
                        for (int j = 0; j < HiddenDim; j++)
                        {
                            gates[g] += _wH[g, j] * hPrev[j];
                        }
                    }

                    iGates[t] = new double[HiddenDim];
                    fGates[t] = new double[HiddenDim];
                    oGates[t] = new double[HiddenDim];
                    gCands[t] = new double[HiddenDim];

                    for (int j = 0; j < HiddenDim; j++)
                    {
                        double ig = Sigmoid(gates[j]);
                        double fg = Sigmoid(gates[HiddenDim + j]);
                        double og = Sigmoid(gates[(2 * HiddenDim) + j]);
                        double gg = Math.Tanh(gates[(3 * HiddenDim) + j]);

                        iGates[t][j] = ig;
                        fGates[t][j] = fg;
                        oGates[t][j] = og;
                        gCands[t][j] = gg;

                        cNext[j] = (fg * cPrev[j]) + (ig * gg);
                        hNext[j] = og * Math.Tanh(cNext[j]);
                    }

                    hHist[t + 1] = hNext;
                    cHist[t + 1] = cNext;
                }

                // Salida y Softmax
                double[] hFinal = hHist[seqLen];
                double[] logits = new double[NumClasses];
                double maxL = double.NegativeInfinity;
                for (int k = 0; k < NumClasses; k++)
                {
                    double sum = _classifierBias[k];
                    for (int j = 0; j < HiddenDim; j++) sum += _classifier[k, j] * hFinal[j];
                    logits[k] = sum;
                    if (sum > maxL) maxL = sum;
                }

                double sumSoftmax = 0.0;
                double[] probs = new double[NumClasses];
                for (int k = 0; k < NumClasses; k++)
                {
                    probs[k] = Math.Exp(logits[k] - maxL);
                    sumSoftmax += probs[k];
                }
                for (int k = 0; k < NumClasses; k++) probs[k] /= Math.Max(1e-9, sumSoftmax);

                // 2. Backward Pass (BPTT)
                double[] dLogits = new double[NumClasses];
                for (int k = 0; k < NumClasses; k++)
                {
                    dLogits[k] = probs[k] - (k == sample.Label ? 1.0 : 0.0);
                }

                double[,] gradCls = new double[NumClasses, HiddenDim];
                double[] gradBiasCls = new double[NumClasses];
                double[] dH = new double[HiddenDim];

                for (int k = 0; k < NumClasses; k++)
                {
                    gradBiasCls[k] = dLogits[k];
                    for (int j = 0; j < HiddenDim; j++)
                    {
                        gradCls[k, j] = dLogits[k] * hFinal[j];
                        dH[j] += dLogits[k] * _classifier[k, j];
                    }
                }

                double[,] gradWX = new double[gatesDim, 1];
                double[,] gradWH = new double[gatesDim, HiddenDim];
                double[] gradB = new double[gatesDim];
                double[] dC = new double[HiddenDim];

                for (int t = seqLen - 1; t >= 0; t--)
                {
                    double x = sample.Values[t];
                    double[] hPrev = hHist[t];
                    double[] cPrev = cHist[t];
                    double[] cCurr = cHist[t + 1];

                    double[] dGates = new double[gatesDim];

                    for (int j = 0; j < HiddenDim; j++)
                    {
                        double ig = iGates[t][j];
                        double fg = fGates[t][j];
                        double og = oGates[t][j];
                        double gg = gCands[t][j];

                        double tanhC = Math.Tanh(cCurr[j]);
                        double dTanhC = 1.0 - (tanhC * tanhC);

                        // Gradiente de compuerta o
                        double dO = dH[j] * tanhC;
                        dGates[(2 * HiddenDim) + j] = dO * og * (1.0 - og);

                        // Acumular gradiente en celda c
                        dC[j] += dH[j] * og * dTanhC;

                        // Gradiente de compuerta f
                        double dF = dC[j] * cPrev[j];
                        dGates[HiddenDim + j] = dF * fg * (1.0 - fg);

                        // Gradiente de compuerta i
                        double dI = dC[j] * gg;
                        dGates[j] = dI * ig * (1.0 - ig);

                        // Gradiente de candidato g
                        double dG = dC[j] * ig;
                        dGates[(3 * HiddenDim) + j] = dG * (1.0 - (gg * gg));
                    }

                    // Acumular gradientes para pesos
                    for (int g = 0; g < gatesDim; g++)
                    {
                        double dg = dGates[g];
                        gradB[g] += dg;
                        gradWX[g, 0] += dg * x;
                        for (int j = 0; j < HiddenDim; j++)
                        {
                            gradWH[g, j] += dg * hPrev[j];
                        }
                    }

                    // Propagación hacia t-1
                    Array.Clear(dH, 0, HiddenDim);
                    for (int j = 0; j < HiddenDim; j++)
                    {
                        for (int g = 0; g < gatesDim; g++)
                        {
                            dH[j] += dGates[g] * _wH[g, j];
                        }
                    }

                    for (int j = 0; j < HiddenDim; j++)
                    {
                        dC[j] *= fGates[t][j];
                    }
                }

                // 3. Actualización Adam
                _adamStep++;
                ApplyAdam(_classifier, gradCls, _mCls, _vCls, lr);
                ApplyAdam1D(_classifierBias, gradBiasCls, _mBias, _vBias, lr);
                ApplyAdam(_wX, gradWX, _mX, _vX, lr);
                ApplyAdam(_wH, gradWH, _mH, _vH, lr);
                ApplyAdam1D(_b, gradB, _mB, _vB, lr);
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

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-Math.Clamp(x, -15.0, 15.0)));

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
