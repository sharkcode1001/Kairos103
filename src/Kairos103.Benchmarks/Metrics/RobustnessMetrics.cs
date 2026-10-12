namespace Kairos103.Benchmarks.Metrics;

/// <summary>
/// Robustness metrics and degradation analysis under time-warping stress.
/// </summary>
public sealed class RobustnessMetrics
{
    public sealed record FactorResult(double WarpFactor, double Accuracy, long OperationsPerInference);

    private readonly List<FactorResult> _results = [];

    public IReadOnlyList<FactorResult> Results => _results;

    public void AddResult(double factor, double accuracy, long opsPerInference)
    {
        _results.Add(new FactorResult(factor, accuracy, opsPerInference));
    }

    /// <summary>
    /// Computes the maximum accuracy degradation relative to the nominal control condition (1.0x).
    /// </summary>
    public double CalculateMaxDegradation()
    {
        var baselineResult = _results.FirstOrDefault(r => Math.Abs(r.WarpFactor - 1.0) < 1e-4);
        if (baselineResult == null) return 0.0;

        double maxDrop = 0.0;
        foreach (var r in _results)
        {
            double drop = baselineResult.Accuracy - r.Accuracy;
            if (drop > maxDrop)
            {
                maxDrop = drop;
            }
        }

        return maxDrop;
    }

    /// <summary>
    /// Checks whether the flat invariance criterion is satisfied (accuracy drop within allowed threshold, e.g. 3%).
    /// </summary>
    public bool MeetsFlatnessCriterion(double maxAllowedDrop = 0.03)
    {
        return CalculateMaxDegradation() <= maxAllowedDrop;
    }
}
