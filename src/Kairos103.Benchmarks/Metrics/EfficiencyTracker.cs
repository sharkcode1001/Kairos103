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

    /// <summary>
    /// Total operations executed across the complete Kairos103 end-to-end pipeline:
    /// Continuous Front-End Transduction MACs + Discrete Spike Events + Classifier Readout Accumulations.
    /// </summary>
    public long TotalKairosOps => TotalFrontEndMacs + TotalSpikes + TotalAccumulateOps;

    public long TotalSamples { get; private set; }
    public long InitialAllocatedBytes { get; private set; }
    public long FinalAllocatedBytes { get; private set; }

    public void RecordSamples(int count) => TotalSamples += count;
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

    /// <summary>
    /// Cumulative heap memory allocated in managed thread during the tracking window.
    /// </summary>
    public long MemoryAllocatedBytes => Math.Max(0, FinalAllocatedBytes - InitialAllocatedBytes);

    /// <summary>
    /// Average managed heap allocation per evaluated inference sample.
    /// </summary>
    public double AllocatedBytesPerSample => TotalSamples > 0 ? (double)MemoryAllocatedBytes / TotalSamples : 0.0;

    /// <summary>
    /// End-to-end computational advantage of Kairos103 over Mini-Transformer:
    /// Ratio = MiniTransformerMacs / TotalKairosOps (FrontEndMacs + Spikes + ReadoutOps).
    /// </summary>
    public double ComputeEfficiencyRatioVsTransformer()
    {
        long kairosTotal = TotalKairosOps;
        return kairosTotal > 0 ? (double)TotalTransformerMacs / kairosTotal : 0.0;
    }

    /// <summary>
    /// End-to-end computational advantage of Kairos103 over Simple LSTM:
    /// Ratio = SimpleLstmMacs / TotalKairosOps (FrontEndMacs + Spikes + ReadoutOps).
    /// </summary>
    public double ComputeEfficiencyRatioVsLstm()
    {
        long kairosTotal = TotalKairosOps;
        return kairosTotal > 0 ? (double)TotalLstmMacs / kairosTotal : 0.0;
    }

    /// <summary>
    /// Default end-to-end efficiency ratio against the dense self-attention baseline (Transformer).
    /// </summary>
    public double ComputeEfficiencyRatio() => ComputeEfficiencyRatioVsTransformer();
}
