using Kairos103.Benchmarks.Metrics;
using Xunit;

namespace Kairos103.Tests;

public class EfficiencyTrackerTests
{
    [Fact]
    public void EfficiencyTracker_EndToEndComputations_IncludeFrontEndAndBaselinesSeparately()
    {
        var tracker = new EfficiencyTracker();

        tracker.RecordFrontEndMacs(60_000_000);
        tracker.RecordSpikes(64_000);
        tracker.RecordAccumulateOps(64_000);

        tracker.RecordTransformerMacs(900_000_000);
        tracker.RecordLstmMacs(150_000_000);

        long expectedKairosTotal = 60_000_000 + 64_000 + 64_000;
        Assert.Equal(expectedKairosTotal, tracker.TotalKairosOps);

        double ratioVsTransformer = tracker.ComputeEfficiencyRatioVsTransformer();
        double ratioVsLstm = tracker.ComputeEfficiencyRatioVsLstm();

        // 900M / ~60.128M =~ 14.96x
        Assert.True(ratioVsTransformer > 14.0 && ratioVsTransformer < 16.0,
            $"Expected ratio vs Transformer ~15x, got {ratioVsTransformer}");

        // 150M / ~60.128M =~ 2.49x
        Assert.True(ratioVsLstm > 2.0 && ratioVsLstm < 3.0,
            $"Expected ratio vs LSTM ~2.5x, got {ratioVsLstm}");

        // ComputeEfficiencyRatio default points to Transformer
        Assert.Equal(ratioVsTransformer, tracker.ComputeEfficiencyRatio());
    }
}
