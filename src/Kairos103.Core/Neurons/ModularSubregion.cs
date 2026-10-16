namespace Kairos103.Core.Neurons;

/// <summary>
/// Modular subregion of N=4 neurons with coupled lateral inhibition.
/// Maps local activation dynamics to a relative arrival order (t1 &lt; t2 &lt; t3 &lt; t4).
/// </summary>
public sealed class ModularSubregion
{
    public const int SubregionSize = 4;
    private readonly LifNeuron[] _neurons;
    private readonly double _lateralInhibitionBoost;

    public ModularSubregion(double baseThreshold = 1.0, double decayRate = 0.95, double lateralInhibitionBoost = 0.25)
    {
        _lateralInhibitionBoost = lateralInhibitionBoost;
        _neurons = new LifNeuron[SubregionSize];
        for (int i = 0; i < SubregionSize; i++)
        {
            _neurons[i] = new LifNeuron(baseThreshold, decayRate);
        }
    }

    public IReadOnlyList<LifNeuron> Neurons => _neurons;

    /// <summary>
    /// Processes a single time step with a vector of 4 current inputs.
    /// If a neuron fires, it raises the threshold of its peers via lateral inhibition.
    /// </summary>
    public void Step(ReadOnlySpan<double> currents, double timestamp)
    {
        if (currents.Length < SubregionSize)
            throw new ArgumentException($"At least {SubregionSize} inputs required.", nameof(currents));

        for (int i = 0; i < SubregionSize; i++)
        {
            if (_neurons[i].Integrate(currents[i], timestamp))
            {
                // Dynamic current-dependent lateral inhibition:
                // Depolarization waves with high amplitude/steep slope exert stronger inhibition on peer neurons,
                // sharpening temporal latency separation without causing combinatorial collapse.
                double dynamicBoost = _lateralInhibitionBoost * (1.0 + Math.Min(2.0, Math.Max(0.0, currents[i] - 1.0)));
                for (int j = 0; j < SubregionSize; j++)
                {
                    if (i != j && !_neurons[j].HasFired)
                    {
                        _neurons[j].Inhibit(dynamicBoost);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Returns the sorted indices of neurons ordered by their first-spike arrival timestamp.
    /// </summary>
    public int[] GetFirstSpikeOrder()
    {
        return Enumerable.Range(0, SubregionSize)
            .OrderBy(i => _neurons[i].FirstSpikeTime)
            .ToArray();
    }

    /// <summary>
    /// Resets all neurons in the subregion, optionally updating the decay rate.
    /// </summary>
    public void Reset(double? dynamicDecayRate = null)
    {
        foreach (var neuron in _neurons)
        {
            neuron.Reset(dynamicDecayRate);
        }
    }
}
