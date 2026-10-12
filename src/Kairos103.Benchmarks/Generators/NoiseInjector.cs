namespace Kairos103.Benchmarks.Generators;

/// <summary>
/// Stochastic perturbation injector: monotonic temporal jitter and Gaussian amplitude noise.
/// </summary>
public sealed class NoiseInjector
{
    private readonly Random _rng;

    public NoiseInjector(int seed = 42)
    {
        _rng = new Random(seed);
    }

    /// <summary>
    /// Generates a normally distributed random variable N(0, stdDev).
    /// </summary>
    public double NextGaussian(double stdDev)
    {
        if (stdDev <= 0.0) return 0.0;
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2) * stdDev;
    }

    /// <summary>
    /// Adds additive Gaussian amplitude noise to the signal.
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
    /// Resamples a signal by applying monotonic temporal jitter across the sampling grid.
    /// </summary>
    public double[] ApplyMonotonicJitter(ReadOnlySpan<double> signal, double jitterStdDev, double amplitudeNoiseStdDev)
    {
        int length = signal.Length;
        double[] grid = new double[length];

        double maxTime = 0.0;
        for (int i = 0; i < length; i++)
        {
            double perturbed = i + NextGaussian(jitterStdDev);
            if (perturbed < maxTime) perturbed = maxTime; // Strict monotonicity
            maxTime = perturbed;
            grid[i] = Math.Clamp(perturbed, 0.0, length - 1.0);
        }

        // Linear interpolation over the jittered grid
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
