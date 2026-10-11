namespace Kairos103.Benchmarks.Metrics;

/// <summary>
/// Métricas de robustez y análisis de degradación frente a dilatación temporal (Time-Warp).
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
    /// Calcula la degradación máxima de precisión con respecto a la condición de control nominal (1.0x).
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
    /// Verifica si se cumple el criterio de invarianza plana (variación menor o igual a un umbral dado, p. ej. 3%).
    /// </summary>
    public bool MeetsFlatnessCriterion(double maxAllowedDrop = 0.03)
    {
        return CalculateMaxDegradation() <= maxAllowedDrop;
    }
}
