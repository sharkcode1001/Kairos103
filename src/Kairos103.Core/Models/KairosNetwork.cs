using Kairos103.Core.Encoding;
using Kairos103.Core.Neurons;

namespace Kairos103.Core.Models;

/// <summary>
/// Red neuronal modular asíncrona Kairos103 con front-end de fase invariante al escalamiento temporal.
/// Compuesta por M subregiones locales de N=4 neuronas LIF con inhibición lateral.
/// </summary>
public sealed class KairosNetwork
{
    public const int FilterPoints = 8;
    public const int TotalPhaseSteps = 120;

    public int SubregionCount { get; }
    public int TotalNeurons => SubregionCount * ModularSubregion.SubregionSize;
    public int TotalFeatures => SubregionCount * 24; // 24 = 4! por subregión

    private readonly ModularSubregion[] _subregions;
    private readonly double[][] _projectionWeights; // Proyección aleatoria fija

    public KairosNetwork(int subregionCount = 16, int seed = 20261010)
    {
        SubregionCount = subregionCount;
        _subregions = new ModularSubregion[subregionCount];
        for (int m = 0; m < subregionCount; m++)
        {
            _subregions[m] = new ModularSubregion(baseThreshold: 0.75, decayRate: 0.90, lateralInhibitionBoost: 0.20);
        }

        // Inicializar proyecciones aleatorias normalizadas fijas
        var rng = new Random(seed);
        _projectionWeights = new double[TotalNeurons][];
        for (int i = 0; i < TotalNeurons; i++)
        {
            _projectionWeights[i] = new double[FilterPoints];
            double sumSq = 0.0;
            for (int k = 0; k < FilterPoints; k++)
            {
                double val = (rng.NextDouble() * 2.0) - 1.0;
                _projectionWeights[i][k] = val;
                sumSq += val * val;
            }

            // Normalización L2
            double norm = Math.Sqrt(sumSq);
            if (norm > 1e-6)
            {
                for (int k = 0; k < FilterPoints; k++)
                {
                    _projectionWeights[i][k] /= norm;
                }
            }
        }
    }

    /// <summary>
    /// Procesa una señal continua 1D mediante un front-end de escala relativa.
    /// Retorna el vector disperso de permutaciones de primer spike, el recuento de spikes y los MACs continuos del front-end.
    /// </summary>
    public (double[] Features, int TotalSpikes, long FrontEndMacs) ForwardDetailed(ReadOnlySpan<double> signal, int effectiveLength)
    {
        // Resetear neuronas
        foreach (var sub in _subregions)
        {
            sub.Reset();
        }

        int length = Math.Min(signal.Length, effectiveLength);
        if (length <= 4)
        {
            return (new double[TotalFeatures], 0, 0);
        }

        int totalSpikes = 0;
        long frontEndMacs = 0;
        Span<double> currents = stackalloc double[ModularSubregion.SubregionSize];

        // Barrido en fase normalizada u en [0, 1]
        for (int step = 0; step < TotalPhaseSteps; step++)
        {
            double phase = (double)step / (TotalPhaseSteps - 1);

            for (int m = 0; m < SubregionCount; m++)
            {
                // Escala de ventana relativa al tamaño de la señal (10% a 28% para estabilidad espectral)
                double scaleFraction = 0.10 + (0.18 * (m % 4) / 3.0);
                double windowSize = Math.Max(2.0, length * scaleFraction);

                double maxStart = Math.Max(0.0, length - windowSize);
                double windowStart = phase * maxStart;

                int baseNeuronIdx = m * ModularSubregion.SubregionSize;

                for (int n = 0; n < ModularSubregion.SubregionSize; n++)
                {
                    double current = 0.0;
                    double[] w = _projectionWeights[baseNeuronIdx + n];

                    for (int k = 0; k < FilterPoints; k++)
                    {
                        double samplePos = windowStart + (k * (windowSize - 1.0) / (FilterPoints - 1.0));
                        int idxLow = (int)Math.Floor(samplePos);
                        int idxHigh = Math.Min(idxLow + 1, length - 1);
                        double frac = samplePos - idxLow;

                        double sampleVal = (idxLow >= 0 && idxLow < length)
                            ? ((1.0 - frac) * signal[idxLow]) + (frac * signal[idxHigh])
                            : 0.0;

                        current += sampleVal * w[k];
                        frontEndMacs++;
                    }

                    currents[n] = Math.Max(0.0, current);
                }

                _subregions[m].Step(currents, timestamp: phase);
            }
        }

        // Construir vector de características por código Lehmer ponderado por rango
        double[] features = new double[TotalFeatures];

        for (int m = 0; m < SubregionCount; m++)
        {
            int firedCount = _subregions[m].Neurons.Count(n => n.HasFired);
            totalSpikes += firedCount;

            if (firedCount >= 2)
            {
                int[] order = _subregions[m].GetFirstSpikeOrder();
                int lehmerIdx = CombinatorialTable.PermutationToLehmerIndex(order);
                int featureOffset = (m * 24) + lehmerIdx;
                // Ponderación proporcional a la fracción de neuronas que dispararon
                features[featureOffset] = (double)firedCount / ModularSubregion.SubregionSize;
            }
        }

        return (features, totalSpikes, frontEndMacs);
    }

    /// <summary>
    /// Sobrecarga compatible hacia atrás.
    /// </summary>
    public (double[] Features, int TotalSpikes) Forward(ReadOnlySpan<double> signal, int effectiveLength)
    {
        var (feats, spikes, _) = ForwardDetailed(signal, effectiveLength);
        return (feats, spikes);
    }
}
