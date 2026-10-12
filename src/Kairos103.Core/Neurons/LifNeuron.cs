namespace Kairos103.Core.Neurons;

/// <summary>
/// Asynchronous Leaky Integrate-and-Fire (LIF) neuron with adaptive dynamic threshold.
/// Records the first-spike arrival timestamp.
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
    /// Integrates an incoming current input at the specified timestamp.
    /// Returns true if an action potential is triggered.
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
    /// Applies adaptive lateral inhibition by transiently raising the firing threshold.
    /// </summary>
    public void Inhibit(double thresholdBoost)
    {
        Threshold += thresholdBoost;
    }

    /// <summary>
    /// Resets the membrane potential and firing state.
    /// Optionally updates the dynamic decay rate.
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
