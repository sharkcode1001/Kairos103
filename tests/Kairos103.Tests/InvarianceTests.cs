using Kairos103.Core.Encoding;
using Xunit;

namespace Kairos103.Tests;

public class InvarianceTests
{
    [Theory]
    [InlineData(0.5, 0.0)]
    [InlineData(0.8, 10.0)]
    [InlineData(1.0, 0.0)]
    [InlineData(1.2, -5.0)]
    [InlineData(2.0, 100.0)]
    [InlineData(5.0, 42.5)]
    public void AffineTransformation_PreservesRelativeSpikeOrder(double alpha, double beta)
    {
        // Tiempos de arribo del primer spike: t1 < t2 < t3 < t4
        double[] nominalArrivalTimes = [12.4, 18.1, 25.6, 33.2];

        // Transformación afín monótona creciente: t' = alpha * t + beta
        double[] warpedArrivalTimes = nominalArrivalTimes
            .Select(t => (alpha * t) + beta)
            .ToArray();

        int[] nominalOrder = RankOrderEncoder.GetPermutation(nominalArrivalTimes);
        int[] warpedOrder = RankOrderEncoder.GetPermutation(warpedArrivalTimes);

        // Teorema 4.1: sigma(T') == sigma(T)
        Assert.Equal(nominalOrder, warpedOrder);

        // Mismo código de Lehmer
        int nominalLehmer = CombinatorialTable.PermutationToLehmerIndex(nominalOrder);
        int warpedLehmer = CombinatorialTable.PermutationToLehmerIndex(warpedOrder);

        Assert.Equal(nominalLehmer, warpedLehmer);
    }

    [Fact]
    public void RankWeights_AreStrictlyIdenticalUnderTimeDilation()
    {
        double[] nominal = [10.0, 20.0, 30.0, 40.0];
        double[] warped = nominal.Select(t => t * 2.5).ToArray();

        double[] weightsNominal = RankOrderEncoder.EncodeRanks(nominal);
        double[] weightsWarped = RankOrderEncoder.EncodeRanks(warped);

        for (int i = 0; i < nominal.Length; i++)
        {
            Assert.Equal(weightsNominal[i], weightsWarped[i], precision: 6);
        }
    }

    [Theory]
    [InlineData(0.5, 0.70)]
    [InlineData(0.8, 0.80)]
    [InlineData(1.2, 0.75)]
    [InlineData(2.0, 0.55)]
    public void KairosNetwork_EndToEnd_PreservesHighFeatureCosineSimilarity_UnderTimeWarp(double warpFactor, double minMeanCosineSim)
    {
        // Prueba de extremo a extremo: pasa señales continuas con dilatación temporal real
        // a través de KairosNetwork (front-end de fase relativa -> subregiones modulares LIF -> Lehmer codes).
        var network = new Kairos103.Core.Models.KairosNetwork(subregionCount: 16);
        double totalSim = 0.0;

        for (int label = 0; label < Kairos103.Benchmarks.Generators.SyntheticSignals.ClassCount; label++)
        {
            double[] nominalSignal = Kairos103.Benchmarks.Generators.SyntheticSignals.GenerateSignal(label, 1.0);
            double[] warpedSignal = Kairos103.Benchmarks.Generators.SyntheticSignals.GenerateSignal(label, warpFactor);

            var (featNominal, _, _) = network.ForwardDetailed(nominalSignal, nominalSignal.Length);
            var (featWarped, _, _) = network.ForwardDetailed(warpedSignal, warpedSignal.Length);

            double dot = 0.0, normNom = 0.0, normWarp = 0.0;
            for (int i = 0; i < featNominal.Length; i++)
            {
                dot += featNominal[i] * featWarped[i];
                normNom += featNominal[i] * featNominal[i];
                normWarp += featWarped[i] * featWarped[i];
            }

            double cosineSim = dot / (Math.Sqrt(normNom) * Math.Sqrt(normWarp) + 1e-12);
            totalSim += cosineSim;
        }

        double meanSim = totalSim / Kairos103.Benchmarks.Generators.SyntheticSignals.ClassCount;

        // La similitud promedio entre características nominales y dilatadas se mantiene robusta
        Assert.True(meanSim >= minMeanCosineSim, $"Mean cosine similarity at warp {warpFactor}x was {meanSim:F3} (expected >= {minMeanCosineSim:F3})");
    }

    [Fact]
    public void KairosNetwork_EndToEnd_ClassifierMaintainsHighAccuracyOnWarpedSignals_WithoutRetraining()
    {
        // Entrena un clasificador exclusivamente a velocidad nominal 1.0x
        var network = new Kairos103.Core.Models.KairosNetwork(subregionCount: 16);
        var trainer = new Kairos103.Core.Models.PairwiseTrainer(Kairos103.Benchmarks.Generators.SyntheticSignals.ClassCount, network.TotalFeatures);

        var trainData = new List<(double[] Features, int Label)>();
        for (int rep = 0; rep < 5; rep++)
        {
            for (int label = 0; label < Kairos103.Benchmarks.Generators.SyntheticSignals.ClassCount; label++)
            {
                double[] sig = Kairos103.Benchmarks.Generators.SyntheticSignals.GenerateSignal(label, 1.0);
                var (feats, _, _) = network.ForwardDetailed(sig, sig.Length);
                trainData.Add((feats, label));
            }
        }

        trainer.Train(trainData, epochs: 15);

        // Evalúa en señales con factores 0.5x, 0.8x, 1.2x, 2.0x sin reentrenar
        double[] factors = [0.5, 0.8, 1.0, 1.2, 2.0];
        int correct = 0;
        int total = 0;

        foreach (double factor in factors)
        {
            for (int label = 0; label < Kairos103.Benchmarks.Generators.SyntheticSignals.ClassCount; label++)
            {
                double[] sig = Kairos103.Benchmarks.Generators.SyntheticSignals.GenerateSignal(label, factor);
                var (feats, _, _) = network.ForwardDetailed(sig, sig.Length);
                var (pred, _) = trainer.Predict(feats);

                if (pred == label) correct++;
                total++;
            }
        }

        double accuracy = (double)correct / total;
        // Invarianza end-to-end: la precisión en señales comprimidas y dilatadas debe ser >= 95%
        Assert.True(accuracy >= 0.95, $"End-to-end classification accuracy on warped signals was {accuracy:P1}");
    }
}

