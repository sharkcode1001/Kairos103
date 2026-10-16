using System.Globalization;
using Kairos103.Core.Data;

namespace Kairos103.Core.Evaluation;

/// <summary>
/// Diagnostic tools for ECG signals, anatomical R-peak alignment, and class template separability.
/// Used to validate alignment and feature efficacy without modifying model weights.
/// </summary>
public static class EcgDiagnostics
{
    private static readonly string[] ClassNames = ["0 (Normal)", "1 (SVEB)", "2 (VEB)", "3 (Fusion)"];

    /// <summary>
    /// Reports the distribution and histogram of detected R-peak positions across samples.
    /// Evaluates whether the peak is centered near sample 125.
    /// </summary>
    public static void ReportRPeakPlacement(IReadOnlyList<KaggleEcgSample> samples)
    {
        if (samples == null || samples.Count == 0) return;

        int n = samples.Count;
        int[] peakPositions = new int[n];
        double sum = 0.0;

        for (int s = 0; s < n; s++)
        {
            var w = samples[s].Waveform;
            int len = Math.Min(w.Length, KaggleEcgSample.WaveformLength);

            // Isoelectric baseline reference (mean of boundaries [0..15] and [len-16..len-1])
            double baseSum = 0.0;
            int baseCount = 0;
            for (int i = 0; i < Math.Min(16, len); i++)
            {
                baseSum += w[i];
                baseCount++;
            }
            for (int i = Math.Max(0, len - 16); i < len; i++)
            {
                baseSum += w[i];
                baseCount++;
            }
            double isoelectric = baseCount > 0 ? baseSum / baseCount : 0.0;

            // Argmax |w[i] - isoelectric|
            int rIdx = 0;
            double maxDev = -1.0;
            for (int i = 0; i < len; i++)
            {
                double dev = Math.Abs(w[i] - isoelectric);
                if (dev > maxDev)
                {
                    maxDev = dev;
                    rIdx = i;
                }
            }

            peakPositions[s] = rIdx;
            sum += rIdx;
        }

        double mean = sum / n;
        double varSum = 0.0;
        for (int s = 0; s < n; s++)
        {
            double diff = peakPositions[s] - mean;
            varSum += diff * diff;
        }
        double std = Math.Sqrt(varSum / n);

        // 25 bins of 10 samples each (0..9, 10..19, ..., 240..249)
        int[] bins = new int[25];
        for (int s = 0; s < n; s++)
        {
            int bin = Math.Clamp(peakPositions[s] / 10, 0, 24);
            bins[bin]++;
        }

        Console.WriteLine("\n==========================================================================");
        Console.WriteLine("          ECG DIAGNOSTICS: R-PEAK PLACEMENT & ALIGNMENT HISTOGRAM         ");
        Console.WriteLine("==========================================================================");
        Console.WriteLine($"Total Analyzed: {n:N0} samples | R-Peak Mean: {mean:F2} (ideal ~125.00) | StdDev: {std:F2}");
        Console.WriteLine("--------------------------------------------------------------------------");
        Console.WriteLine(" Bin Range   | Count   | Pct    | Distribution Bar");
        Console.WriteLine("--------------------------------------------------------------------------");

        int maxBinCount = bins.Max();
        for (int b = 0; b < 25; b++)
        {
            int start = b * 10;
            int end = start + 9;
            double pct = (double)bins[b] / n * 100.0;
            int barLen = maxBinCount > 0 ? (int)Math.Round((double)bins[b] / maxBinCount * 35.0) : 0;
            string bar = new('#', barLen);
            string marker = (start <= 125 && 125 <= end) ? " <-- Center [120-129]" : "";
            Console.WriteLine($" [{start,3} - {end,3}] | {bins[b],7:N0} | {pct,5:F1}% | {bar}{marker}");
        }
        Console.WriteLine("==========================================================================\n");
    }

    /// <summary>
    /// Computes and reports Pearson correlation of each class's QRS morphology against the Class 0 (Normal) template.
    /// Verifies if Normal ~ 1.0, VEB is low, and Fusion is intermediate (0.4 - 0.7).
    /// </summary>
    public static void ReportTemplateSeparability(IReadOnlyList<KaggleEcgSample> samples)
    {
        if (samples == null || samples.Count == 0) return;

        const int qrsStart = 85;
        const int qrsEnd = 165;
        const int qrsLen = qrsEnd - qrsStart; // 80 points

        // 1. Build average QRS template for Class 0 (min-max normalized per QRS)
        double[] template0 = new double[qrsLen];
        int class0Count = 0;

        for (int s = 0; s < samples.Count; s++)
        {
            if (samples[s].Label != 0) continue;

            double[] normQrs = ExtractMinMaxQrs(samples[s].Waveform, qrsStart, qrsEnd);
            for (int k = 0; k < qrsLen; k++)
            {
                template0[k] += normQrs[k];
            }
            class0Count++;
        }

        if (class0Count == 0)
        {
            Console.WriteLine("[WARN] No Class 0 samples found to construct reference QRS template.");
            return;
        }

        for (int k = 0; k < qrsLen; k++)
        {
            template0[k] /= class0Count;
        }

        // 2. Compute Pearson correlation of each sample against Class 0 template
        var classCorrs = new List<double>[4];
        for (int c = 0; c < 4; c++) classCorrs[c] = [];

        for (int s = 0; s < samples.Count; s++)
        {
            int label = samples[s].Label;
            if (label < 0 || label >= 4) continue;

            double[] normQrs = ExtractMinMaxQrs(samples[s].Waveform, qrsStart, qrsEnd);
            double r = ComputePearsonCorrelation(normQrs, template0);
            classCorrs[label].Add(r);
        }

        Console.WriteLine("==========================================================================");
        Console.WriteLine("     ECG DIAGNOSTICS: TEMPLATE SEPARABILITY (PEARSON r vs. NORMAL QRS)    ");
        Console.WriteLine("==========================================================================");
        Console.WriteLine(" Class Name     | Count   | Pearson r (Mean ± Std) | Min r   | Max r");
        Console.WriteLine("--------------------------------------------------------------------------");

        for (int c = 0; c < 4; c++)
        {
            var list = classCorrs[c];
            if (list.Count == 0)
            {
                Console.WriteLine($" {ClassNames[c],-14} | {0,7} | N/A                    | N/A     | N/A");
                continue;
            }

            double meanR = list.Average();
            double varR = list.Select(v => (v - meanR) * (v - meanR)).Sum() / list.Count;
            double stdR = Math.Sqrt(varR);
            double minR = list.Min();
            double maxR = list.Max();

            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                " {0,-14} | {1,7:N0} | {2,6:F4} ± {3,6:F4}        | {4,7:F4} | {5,7:F4}",
                ClassNames[c], list.Count, meanR, stdR, minR, maxR));
        }
        Console.WriteLine("--------------------------------------------------------------------------");
        Console.WriteLine(" Clinical Diagnostic Reference:");
        Console.WriteLine("   Class 0 (Normal): Expected r >= 0.85 - 0.95 (high self-correlation)");
        Console.WriteLine("   Class 1 (SVEB):   Expected r >= 0.80 - 0.92 (ventricles normal, QRS preserved)");
        Console.WriteLine("   Class 2 (VEB):    Expected r <= 0.30 - 0.50 (wide, bizarre, ectopic morphology)");
        Console.WriteLine("   Class 3 (Fusion): Expected r ~  0.45 - 0.75 (hybrid intermediate morphology)");
        Console.WriteLine("==========================================================================\n");
    }

    private static double[] ExtractMinMaxQrs(double[] waveform, int start, int end)
    {
        int len = end - start;
        double[] qrs = new double[len];
        int wLen = waveform.Length;

        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        for (int i = 0; i < len; i++)
        {
            int srcIdx = start + i;
            double val = (srcIdx >= 0 && srcIdx < wLen) ? waveform[srcIdx] : 0.0;
            qrs[i] = val;
            if (val < min) min = val;
            if (val > max) max = val;
        }

        double p2p = max - min;
        double invP2p = p2p > 1e-6 ? 1.0 / p2p : 1.0;
        double offset = p2p > 1e-6 ? min : 0.0;

        for (int i = 0; i < len; i++)
        {
            qrs[i] = (qrs[i] - offset) * invP2p;
        }

        return qrs;
    }

    private static double ComputePearsonCorrelation(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        int n = Math.Min(x.Length, y.Length);
        if (n <= 1) return 0.0;

        double sumX = 0.0, sumY = 0.0;
        for (int i = 0; i < n; i++)
        {
            sumX += x[i];
            sumY += y[i];
        }
        double meanX = sumX / n;
        double meanY = sumY / n;

        double num = 0.0, denomX = 0.0, denomY = 0.0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - meanX;
            double dy = y[i] - meanY;
            num += dx * dy;
            denomX += dx * dx;
            denomY += dy * dy;
        }

        double denom = Math.Sqrt(denomX * denomY);
        return denom > 1e-8 ? num / denom : 0.0;
    }
}
