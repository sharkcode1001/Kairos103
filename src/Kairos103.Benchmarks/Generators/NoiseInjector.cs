namespace Kairos103.Benchmarks.Generators;

/// <summary>
/// Inyector de perturbaciones estocásticas: jitter temporal monótono y ruido de amplitud gaussiano.
/// </summary>
public sealed class NoiseInjector
{
    private readonly Random _rng;

    public NoiseInjector(int seed = 42)
    {
        _rng = new Random(seed);
    }

    /// <summary>
    /// Genera una variable aleatoria con distribución normal N(0, stdDev).
    /// </summary>
    public double NextGaussian(double stdDev)
    {
        if (stdDev <= 0.0) return 0.0;
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2) * stdDev;
    }

    /// <summary>
    /// Aplica ruido de amplitud aditivo gaussiano.
    /// </summary>
    public double[] AddAmplitudeNoise(ReadOnlySpan<double> signal, double stdDev)
    {
        double[] noisy = new double[signal.Length];
        for (int i = 0; i < signal.Length; i++)
        {
            noisy[i] = signal[i] + NextGaussian(stdDev);
        }
        return noisy;
    }

    /// <summary>
    /// Remuestrea una señal agregando jitter temporal monótono a la grilla de muestreo.
    /// </summary>
    public double[] ApplyMonotonicJitter(ReadOnlySpan<double> signal, double jitterStdDev, double amplitudeNoiseStdDev)
    {
        int length = signal.Length;
        double[] grid = new double[length];

        double maxTime = 0.0;
        for (int i = 0; i < length; i++)
        {
            double perturbed = i + NextGaussian(jitterStdDev);
            if (perturbed < maxTime) perturbed = maxTime; // Monotonía estricta
            maxTime = perturbed;
            grid[i] = Math.Clamp(perturbed, 0.0, length - 1.0);
        }

        // Interpolación lineal sobre la grilla con jitter
        double[] result = new double[length];
        for (int i = 0; i < length; i++)
        {
            double pos = grid[i];
            int low = (int)Math.Floor(pos);
            int high = Math.Min(low + 1, length - 1);
            double frac = pos - low;

            double interp = ((1.0 - frac) * signal[low]) + (frac * signal[high]);
            result[i] = interp + NextGaussian(amplitudeNoiseStdDev);
        }

        return result;
    }
}
