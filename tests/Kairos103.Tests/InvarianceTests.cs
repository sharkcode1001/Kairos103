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
}
