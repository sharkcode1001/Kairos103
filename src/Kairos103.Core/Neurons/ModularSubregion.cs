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
                // Lateral inhibition to remaining neurons in subregion
                for (int j = 0; j < SubregionSize; j++)
                {
                    if (i != j && !_neurons[j].HasFired)
                    {
                        _neurons[j].Inhibit(_lateralInhibitionBoost);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Returns the sorted indices of neurons ordered by their first-spike arrival timestamp.
    /// Stable with respect to ties (equal timestamps preserve ascending index order).
    /// </summary>
    public int[] GetFirstSpikeOrder()
    {
        int[] order = new int[SubregionSize];
        GetFirstSpikeOrder(order);
        return order;
    }

    /// <summary>
    /// Fills <paramref name="order"/> with the neuron indices ordered by first-spike arrival timestamp.
    /// Allocation-free stable insertion sort (equivalent to OrderBy, which is stable).
    /// </summary>
    public void GetFirstSpikeOrder(Span<int> order)
    {
        if (order.Length < SubregionSize)
            throw new ArgumentException($"At least {SubregionSize} slots required.", nameof(order));

        for (int i = 0; i < SubregionSize; i++) order[i] = i;

        for (int i = 1; i < SubregionSize; i++)
        {
            int key = order[i];
            double keyTime = _neurons[key].FirstSpikeTime;
            int j = i - 1;
            while (j >= 0 && _neurons[order[j]].FirstSpikeTime > keyTime)
            {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }
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
