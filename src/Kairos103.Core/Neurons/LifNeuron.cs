namespace Kairos103.Core.Neurons;

/// <summary>
/// Neurona Leaky Integrate-and-Fire (LIF) asíncrona con umbral adaptativo.
/// Registra el instante de primer disparo (first-spike arrival time).
/// </summary>
public sealed class LifNeuron
{
    public double MembranePotential { get; private set; }
    public double Threshold { get; private set; }
    public double DecayRate { get; set; }
    public double FirstSpikeTime { get; private set; } = double.PositiveInfinity;
    public bool HasFired => !double.IsPositiveInfinity(FirstSpikeTime);

    private readonly double _baseThreshold;

    public LifNeuron(double threshold = 1.0, double decayRate = 0.95)
    {
        _baseThreshold = threshold;
        Threshold = threshold;
        DecayRate = decayRate;
        Reset();
    }

    /// <summary>
    /// Integra una corriente de entrada en el instante t.
    /// Retorna true si dispara un potencial de acción.
    /// </summary>
    public bool Integrate(double current, double timestamp)
    {
        MembranePotential = (MembranePotential * DecayRate) + current;

        if (MembranePotential >= Threshold && !HasFired)
        {
            FirstSpikeTime = timestamp;
            MembranePotential = 0.0;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Aplica inhibición lateral adaptativa incrementando transitoriamente el umbral.
    /// </summary>
    public void Inhibit(double thresholdBoost)
    {
        Threshold += thresholdBoost;
    }

    /// <summary>
    /// Restablece el potencial de membrana y el estado de disparo.
    /// Opcionalmente actualiza la tasa de decaimiento dinámica.
    /// </summary>
    public void Reset(double? dynamicDecayRate = null)
    {
        if (dynamicDecayRate.HasValue)
        {
            DecayRate = dynamicDecayRate.Value;
        }
        MembranePotential = 0.0;
        Threshold = _baseThreshold;
        FirstSpikeTime = double.PositiveInfinity;
    }
}
