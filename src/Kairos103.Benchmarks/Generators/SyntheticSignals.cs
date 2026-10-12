namespace Kairos103.Benchmarks.Generators;

/// <summary>
/// Programmatic C# generator of 1D synthetic signals with time-warp scaling.
/// Implements the waveform patterns from the WarpStress v1 benchmark.
/// </summary>
public sealed class SyntheticSignals
{
    public const int BaseLength = 128;
    public const int ClassCount = 4;

    private static double Gauss(double t, double center, double width)
    {
        double d = (t - center) / width;
        return Math.Exp(-(d * d));
    }

    private static double Pulse(double t, double freq, double center, double width)
    {
        return Math.Sin(2.0 * Math.PI * freq * (t - center)) * Gauss(t, center, width);
    }

    private static double Chirp(double t, double f0, double f1, double t0 = 0.15, double t1 = 0.85, double center = 0.5, double width = 0.30)
    {
        double tau = Math.Clamp(t, t0, t1) - t0;
        double phase = (f0 * tau) + ((f1 - f0) * tau * tau / (2.0 * (t1 - t0)));
        return Math.Sin(2.0 * Math.PI * phase) * Gauss(t, center, width);
    }

    public static double EvaluateTemplate(int label, double t)
    {
        return label switch
        {
            0 => Pulse(t, 4.0, 0.28, 0.09) + Pulse(t, 13.0, 0.72, 0.09),
            1 => Pulse(t, 13.0, 0.28, 0.09) + Pulse(t, 4.0, 0.72, 0.09),
            2 => Chirp(t, 3.0, 15.0),
            3 => Chirp(t, 15.0, 3.0),
            _ => throw new ArgumentOutOfRangeException(nameof(label))
        };
    }

    /// <summary>
    /// Generates a signal sample resampled at a specific speed scale factor (alpha).
    /// </summary>
    public static double[] GenerateSignal(int label, double warpFactor)
    {
        int length = (int)Math.Round(BaseLength / warpFactor);
        double[] signal = new double[length];

        for (int i = 0; i < length; i++)
        {
            double t = (i * warpFactor) / BaseLength;
            signal[i] = EvaluateTemplate(label, Math.Clamp(t, 0.0, 1.0));
        }

        return signal;
    }
}
