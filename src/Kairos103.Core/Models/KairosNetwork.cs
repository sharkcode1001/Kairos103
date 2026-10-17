using System.Numerics;
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
    private readonly double[] _projectionWeights; // Fixed normalized random projection, flattened (TotalNeurons x FilterPoints) for cache/SIMD friendliness
    private readonly FilterCoord[]? _coords250;

    public readonly struct FilterCoord
    {
        public readonly int IdxLow;
        public readonly int IdxHigh;
        public readonly double Frac;

        public FilterCoord(int low, int high, double frac)
        {
            IdxLow = low;
            IdxHigh = high;
            Frac = frac;
        }
    }

    public KairosNetwork(int subregionCount = 16, int seed = 20261010)
    {
        SubregionCount = subregionCount;
        _subregions = new ModularSubregion[subregionCount];
        for (int m = 0; m < subregionCount; m++)
        {
            _subregions[m] = new ModularSubregion(baseThreshold: 0.75, decayRate: 0.90, lateralInhibitionBoost: 0.20);
        }

        // Initialize fixed normalized random projections (flattened contiguous layout).
        // The RNG call order is preserved to keep seeded weights bit-identical to the previous jagged layout.
        var rng = new Random(seed);
        _projectionWeights = new double[TotalNeurons * FilterPoints];
        for (int i = 0; i < TotalNeurons; i++)
        {
            int offset = i * FilterPoints;
            double sumSq = 0.0;
            for (int k = 0; k < FilterPoints; k++)
            {
                double val = (rng.NextDouble() * 2.0) - 1.0;
                _projectionWeights[offset + k] = val;
                sumSq += val * val;
            }

            // L2 normalization
            double norm = Math.Sqrt(sumSq);
            if (norm > 1e-6)
            {
                for (int k = 0; k < FilterPoints; k++)
                {
                    _projectionWeights[offset + k] /= norm;
                }
            }
        }

        _coords250 = PrecomputeCoordinates(subregionCount, 250);
    }

    private KairosNetwork(int subregionCount, double[] projectionWeights, FilterCoord[]? coords250)
    {
        SubregionCount = subregionCount;
        _projectionWeights = projectionWeights;
        _coords250 = coords250;
        _subregions = new ModularSubregion[subregionCount];
        for (int m = 0; m < subregionCount; m++)
        {
            _subregions[m] = new ModularSubregion(baseThreshold: 0.75, decayRate: 0.90, lateralInhibitionBoost: 0.20);
        }
    }

    /// <summary>
    /// Creates an independent clone of the network sharing read-only weights and caches.
    /// Thread-safe for parallel signal encoding.
    /// </summary>
    public KairosNetwork Clone() => new KairosNetwork(SubregionCount, _projectionWeights, _coords250);

    /// <summary>
    /// SIMD-accelerated dot product over exactly <see cref="FilterPoints"/> elements.
    /// Falls back to a scalar tail loop for architectures whose vector width does not divide the length.
    /// </summary>
    private static double DotProduct(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        int width = Vector<double>.Count;
        Vector<double> acc = Vector<double>.Zero;
        int i = 0;
        for (; i <= FilterPoints - width; i += width)
        {
            acc += new Vector<double>(a.Slice(i, width)) * new Vector<double>(b.Slice(i, width));
        }

        double sum = Vector.Dot(acc, Vector<double>.One);
        for (; i < FilterPoints; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static FilterCoord[] PrecomputeCoordinates(int subregionCount, int length)
    {
        var coords = new FilterCoord[TotalPhaseSteps * subregionCount * FilterPoints];
        for (int step = 0; step < TotalPhaseSteps; step++)
        {
            double phase = (double)step / (TotalPhaseSteps - 1);
            for (int m = 0; m < subregionCount; m++)
            {
                // Multi-scale temporal receptive fields across 4 specialized anatomical banks
                int bank = m % 4;
                int scaleIdx = (m / 4) % 4;
                double scaleFraction = 0.06 + (0.12 * (scaleIdx / 3.0));
                double windowSize = Math.Max(2.0, length * scaleFraction);

                // Temporal receptive field banks:
                // Bank 0: Global sweep [0.00, 1.00] - captures rhythm and broad morphology
                // Bank 1: P-wave / PR interval [0.08, 0.36] - critical for SVEB (atrial depolarization)
                // Bank 2: Central QRS complex [0.32, 0.66] - critical for VEB and Fusion (ventricular depolarization)
                // Bank 3: ST segment / T-wave [0.58, 0.90] - critical for repolarization discordance
                double minPhase = bank switch
                {
                    1 => 0.08,
                    2 => 0.32,
                    3 => 0.58,
                    _ => 0.00
                };
                double maxPhase = bank switch
                {
                    1 => 0.36,
                    2 => 0.66,
                    3 => 0.90,
                    _ => 1.00
                };

                double phaseAdjusted = minPhase + (phase * (maxPhase - minPhase));
                double maxStart = Math.Max(0.0, length - windowSize);
                double windowStart = Math.Clamp(phaseAdjusted * maxStart, 0.0, maxStart);

                int offset = ((step * subregionCount) + m) * FilterPoints;
                for (int k = 0; k < FilterPoints; k++)
                {
                    double samplePos = windowStart + (k * (windowSize - 1.0) / (FilterPoints - 1.0));
                    int idxLow = (int)Math.Floor(samplePos);
                    int idxHigh = Math.Min(idxLow + 1, length - 1);
                    double frac = samplePos - idxLow;
                    coords[offset + k] = new FilterCoord(idxLow, idxHigh, frac);
                }
            }
        }
        return coords;
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
        Span<double> sampledPoints = stackalloc double[FilterPoints];
        bool useFastCoords = (length == 250 && _coords250 != null && signal.Length >= 250);

        // Sweep in normalized phase u in [0, 1]
        for (int step = 0; step < TotalPhaseSteps; step++)
        {
            double phase = (double)step / (TotalPhaseSteps - 1);

            for (int m = 0; m < SubregionCount; m++)
            {
                int baseNeuronIdx = m * ModularSubregion.SubregionSize;

                if (useFastCoords)
                {
                    int coordOffset = ((step * SubregionCount) + m) * FilterPoints;
                    for (int k = 0; k < FilterPoints; k++)
                    {
                        var c = _coords250![coordOffset + k];
                        sampledPoints[k] = ((1.0 - c.Frac) * signal[c.IdxLow]) + (c.Frac * signal[c.IdxHigh]);
                    }
                }
                else
                {
                    int bank = m % 4;
                    int scaleIdx = (m / 4) % 4;
                    double scaleFraction = 0.06 + (0.12 * (scaleIdx / 3.0));
                    double windowSize = Math.Max(2.0, length * scaleFraction);

                    double minPhase = bank switch
                    {
                        1 => 0.08,
                        2 => 0.32,
                        3 => 0.58,
                        _ => 0.00
                    };
                    double maxPhase = bank switch
                    {
                        1 => 0.36,
                        2 => 0.66,
                        3 => 0.90,
                        _ => 1.00
                    };

                    double phaseAdjusted = minPhase + (phase * (maxPhase - minPhase));
                    double maxStart = Math.Max(0.0, length - windowSize);
                    double windowStart = Math.Clamp(phaseAdjusted * maxStart, 0.0, maxStart);

                    for (int k = 0; k < FilterPoints; k++)
                    {
                        double samplePos = windowStart + (k * (windowSize - 1.0) / (FilterPoints - 1.0));
                        int idxLow = (int)Math.Floor(samplePos);
                        int idxHigh = Math.Min(idxLow + 1, length - 1);
                        double frac = samplePos - idxLow;

                        sampledPoints[k] = (idxLow >= 0 && idxLow < length)
                            ? ((1.0 - frac) * signal[idxLow]) + (frac * signal[idxHigh])
                            : 0.0;
                    }
                }

                for (int n = 0; n < ModularSubregion.SubregionSize; n++)
                {
                    var w = _projectionWeights.AsSpan((baseNeuronIdx + n) * FilterPoints, FilterPoints);
                    double current = DotProduct(sampledPoints, w);
                    frontEndMacs += FilterPoints;
                    currents[n] = Math.Max(0.0, current);
                }

                _subregions[m].Step(currents, timestamp: phase);
            }
        }

        // Build feature vector via Lehmer code weighted by firing fraction
        double[] features = new double[TotalFeatures];
        Span<int> order = stackalloc int[ModularSubregion.SubregionSize];

        for (int m = 0; m < SubregionCount; m++)
        {
            int firedCount = 0;
            var neurons = _subregions[m].Neurons;
            for (int n = 0; n < neurons.Count; n++)
            {
                if (neurons[n].HasFired) firedCount++;
            }
            totalSpikes += firedCount;

            if (firedCount >= 2)
            {
                _subregions[m].GetFirstSpikeOrder(order);
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
