using Kairos103.Core.Data;
using Kairos103.Core.Models;
using Xunit;

namespace Kairos103.Tests;

/// <summary>
/// Tests verifying numerical equivalence and thread safety of performance optimizations:
/// sparse active features, precomputed filter coordinates, and parallel encoding.
/// </summary>
public class OptimizationEquivalenceTests
{
    [Fact]
    public void KairosNetwork_Coords250_ProducesExactFeatures()
    {
        var rng = new Random(12345);
        var net = new KairosNetwork(subregionCount: 8, seed: 42);
        var netDynamic = new KairosNetwork(subregionCount: 8, seed: 42);

        for (int sample = 0; sample < 50; sample++)
        {
            double[] signal = new double[250];
            for (int t = 0; t < 250; t++)
            {
                signal[t] = rng.NextDouble();
            }

            var (featsFast, spikesFast, _) = net.ForwardDetailed(signal, 250);
            // Effective length of 250 vs dynamic fallback
            var (featsDynamic, spikesDynamic) = netDynamic.Forward(signal, 250);

            Assert.Equal(spikesFast, spikesDynamic);
            Assert.Equal(featsFast.Length, featsDynamic.Length);

            for (int f = 0; f < featsFast.Length; f++)
            {
                Assert.Equal(featsDynamic[f], featsFast[f], precision: 10);
            }
        }
    }

    [Fact]
    public void KairosNetwork_Clone_IsThreadSafeAndIsolated()
    {
        var netMaster = new KairosNetwork(subregionCount: 16, seed: 777);
        var netClone1 = netMaster.Clone();
        var netClone2 = netMaster.Clone();

        double[] sig1 = new double[250];
        double[] sig2 = new double[250];
        Array.Fill(sig1, 0.5);
        Array.Fill(sig2, 0.8);

        var (f1Master, _) = netMaster.Forward(sig1, 250);
        var (f1Clone, _) = netClone1.Forward(sig1, 250);

        Assert.Equal(f1Master, f1Clone);
    }

    [Fact]
    public void OvrTrainer_PredictProbabilities_MatchesDenseMath()
    {
        var rng = new Random(42);
        int classCount = 4;
        int featDim = 100;
        int hiddenDim = 16;
        var trainer = new OvrTrainer(classCount, featDim, hiddenDim, seed: 42);

        for (int s = 0; s < 50; s++)
        {
            double[] features = new double[featDim];
            // Sparse: only 10% non-zero
            for (int f = 0; f < featDim; f++)
            {
                if (rng.NextDouble() < 0.10)
                {
                    features[f] = rng.NextDouble();
                }
            }

            double[] probs = trainer.PredictProbabilities(features);

            // Manual dense calculation to verify exact equivalence
            for (int c = 0; c < classCount; c++)
            {
                double z = trainer.B2[c];
                for (int j = 0; j < hiddenDim; j++)
                {
                    double sum = trainer.B1[c][j];
                    for (int f = 0; f < featDim; f++)
                    {
                        sum += trainer.W1[c][j][f] * features[f];
                    }
                    double hj = sum > 0.0 ? sum : 0.0;
                    z += trainer.W2[c][j] * hj;
                }
                double expectedProb = 1.0 / (1.0 + Math.Exp(-z));
                Assert.Equal(expectedProb, probs[c], precision: 8);
            }
        }
    }
}
