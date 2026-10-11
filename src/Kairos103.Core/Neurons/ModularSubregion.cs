namespace Kairos103.Core.Neurons;

/// <summary>
/// Subregión modular de N=4 neuronas con inhibición lateral acoplada.
/// Mapea la dinámica de activación local a un orden relativo de llegada (t1 &lt; t2 &lt; t3 &lt; t4).
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
    /// Procesa un paso temporal con un vector de 4 entradas de corriente.
    /// Si una neurona dispara, eleva el umbral de sus compañeras mediante inhibición lateral.
    /// </summary>
    public void Step(ReadOnlySpan<double> currents, double timestamp)
    {
        if (currents.Length < SubregionSize)
            throw new ArgumentException($"Se requieren al menos {SubregionSize} entradas.", nameof(currents));

        for (int i = 0; i < SubregionSize; i++)
        {
            if (_neurons[i].Integrate(currents[i], timestamp))
            {
                // Inhibición lateral al resto de neuronas en la subregión
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
    /// Retorna los índices ordenados de las neuronas según su instante de primer disparo.
    /// </summary>
    public int[] GetFirstSpikeOrder()
    {
        return Enumerable.Range(0, SubregionSize)
            .OrderBy(i => _neurons[i].FirstSpikeTime)
            .ToArray();
    }

    /// <summary>
    /// Restablece todas las neuronas de la subregión, opcionalmente actualizando la tasa de decaimiento.
    /// </summary>
    public void Reset(double? dynamicDecayRate = null)
    {
        foreach (var neuron in _neurons)
        {
            neuron.Reset(dynamicDecayRate);
        }
    }
}
