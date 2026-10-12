namespace Kairos103.Benchmarks.Metrics;

/// <summary>
/// Computational efficiency and memory allocation tracker.
/// Compares discrete spike events and front-end MACs against continuous baseline multiply-accumulate operations (MACs).
/// </summary>
public sealed class EfficiencyTracker
{
    public long TotalSpikes { get; private set; }
    public long TotalAccumulateOps { get; private set; }
    public long TotalFrontEndMacs { get; private set; }
    public long TotalTransformerMacs { get; private set; }
    public long TotalLstmMacs { get; private set; }
    public long TotalContinuousMacs => TotalTransformerMacs + TotalLstmMacs;
    public long InitialAllocatedBytes { get; private set; }
    public long FinalAllocatedBytes { get; private set; }

    public void RecordSpikes(int count) => TotalSpikes += count;
    public void RecordAccumulateOps(int count) => TotalAccumulateOps += count;
    public void RecordFrontEndMacs(long count) => TotalFrontEndMacs += count;
    public void RecordTransformerMacs(long count) => TotalTransformerMacs += count;
    public void RecordLstmMacs(long count) => TotalLstmMacs += count;
    public void RecordMacs(long count) => TotalTransformerMacs += count;

    public void StartMemoryTracking()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        InitialAllocatedBytes = GC.GetAllocatedBytesForCurrentThread();
    }

    public void StopMemoryTracking()
    {
        FinalAllocatedBytes = GC.GetAllocatedBytesForCurrentThread();
    }

    public long MemoryAllocatedBytes => Math.Max(0, FinalAllocatedBytes - InitialAllocatedBytes);

    public double ComputeEfficiencyRatio()
    {
        long discreteTotal = TotalSpikes + TotalAccumulateOps;
        return discreteTotal > 0 ? (double)TotalContinuousMacs / discreteTotal : 0.0;
    }
}
