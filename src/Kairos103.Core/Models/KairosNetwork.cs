using Kairos103.Core.Encoding;
using Kairos103.Core.Neurons;

namespace Kairos103.Core.Models;

/// <summary>
/// Modular asynchronous neural network Kairos103 with time-warp invariant phase front-end.
/// Composed of M local subregions of N=4 LIF neurons with lateral inhibition.
/// </summary>
public sealed class KairosNetwork
{
    public const int FilterPoints = 8;
    public const int TotalPhaseSteps = 120;

    public int SubregionCount { get; }
    public int TotalNeurons => SubregionCount * ModularSubregion.SubregionSize;
    public int TotalFeatures => SubregionCount * 24; // 24 = 4! per subregion

    private readonly ModularSubregion[] _subregions;
    private readonly double[][] _projectionWeights; // Fixed normalized random projection

    public KairosNetwork(int subregionCount = 16, int seed = 20261010)
    {
        SubregionCount = subregionCount;
        _subregions = new ModularSubregion[subregionCount];
        for (int m = 0; m < subregionCount; m++)
        {
            _subregions[m] = new ModularSubregion(baseThreshold: 0.75, decayRate: 0.90, lateralInhibitionBoost: 0.20);
        }

        // Initialize fixed normalized random projections
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

            // L2 normalization
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
    /// Processes a continuous 1D signal through the relative-scale front-end.
    /// Returns the sparse first-spike permutation feature vector, total spike count, and front-end continuous MACs.
    /// </summary>
    public (double[] Features, int TotalSpikes, long FrontEndMacs) ForwardDetailed(ReadOnlySpan<double> signal, int effectiveLength)
    {
        // Reset neurons
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

        // Sweep in normalized phase u in [0, 1]
        for (int step = 0; step < TotalPhaseSteps; step++)
        {
            double phase = (double)step / (TotalPhaseSteps - 1);

            for (int m = 0; m < SubregionCount; m++)
            {
                // Window scale relative to signal length (10% to 28% for spectral stability)
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

        // Build feature vector via Lehmer code weighted by firing fraction
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
                // Weighting proportional to the fraction of fired neurons
                features[featureOffset] = (double)firedCount / ModularSubregion.SubregionSize;
            }
        }

        return (features, totalSpikes, frontEndMacs);
    }

    /// <summary>
    /// Backward-compatible overload.
    /// </summary>
    public (double[] Features, int TotalSpikes) Forward(ReadOnlySpan<double> signal, int effectiveLength)
    {
        var (feats, spikes, _) = ForwardDetailed(signal, effectiveLength);
        return (feats, spikes);
    }
}
