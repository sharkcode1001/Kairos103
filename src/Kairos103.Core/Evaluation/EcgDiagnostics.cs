using System.Globalization;
using Kairos103.Core.Data;
using Kairos103.Core.Models;

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

    /// <summary>
    /// Estimates the contribution of each auxiliary feature using permutation importance (a model-agnostic ablation):
    /// each auxiliary column is independently shuffled across the evaluation set and the induced drop in Macro F1 is measured.
    /// It also reports distributional health (min/max/mean/std) and an eta-squared separability score to flag dead or redundant features.
    /// </summary>
    public static void ReportAuxFeatureImportance(
        KairosClassifier model,
        IReadOnlyList<(double[] Features, int Label)> encoded,
        int repeats = 3,
        int seed = 12345)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (encoded == null || encoded.Count == 0) return;

        int auxCount = KaggleEcgSample.AuxCount;
        int n = encoded.Count;
        int featDim = encoded[0].Features.Length;
        if (featDim <= auxCount || model.TotalFeatures != featDim)
        {
            Console.WriteLine($"[WARN] Feature importance skipped: encoded feature dim ({featDim}) does not match the model ({model.TotalFeatures}) with {auxCount} aux slots.");
            return;
        }

        int auxStart = featDim - auxCount;
        int classCount = model.ClassCount;

        int[] actuals = new int[n];
        for (int i = 0; i < n; i++) actuals[i] = encoded[i].Label;

        bool useBias = model.DecisionBias is { Length: > 0 } && model.DecisionBias.Length == classCount;
        bool useThresh = model.Thresholds is { Length: > 0 } && model.Thresholds.Length == classCount;

        int Predict(double[] feats)
        {
            if (useBias)
            {
                return KairosClassifier.ArgMaxBias(model.PredictCalibratedLogitsFromFeatures(feats), model.DecisionBias);
            }
            if (useThresh)
            {
                double[] probs = model.PredictProbabilitiesFromFeatures(feats);
                int best = 0;
                double bestScore = probs[0] / Math.Max(1e-4, model.Thresholds[0]);
                for (int c = 1; c < classCount; c++)
                {
                    double score = probs[c] / Math.Max(1e-4, model.Thresholds[c]);
                    if (score > bestScore) { bestScore = score; best = c; }
                }
                return best;
            }
            return KairosClassifier.ArgMaxBias(model.PredictCalibratedLogitsFromFeatures(feats), new double[classCount]);
        }

        int[] basePreds = new int[n];
        for (int i = 0; i < n; i++) basePreds[i] = Predict(encoded[i].Features);
        double baseF1 = MacroF1Metrics.Compute(actuals, basePreds, classCount).MacroF1;

        var rng = new Random(seed);
        var rows = new (int Index, string Name, double Min, double Max, double Mean, double Std, double Eta2, double Drop)[auxCount];

        for (int a = 0; a < auxCount; a++)
        {
            int idx = auxStart + a;
            double[] col = new double[n];
            double mn = double.PositiveInfinity, mx = double.NegativeInfinity, sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                double v = encoded[i].Features[idx];
                col[i] = v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
            }
            double mean = sum / n;

            double totalSs = 0.0;
            for (int i = 0; i < n; i++) { double d = col[i] - mean; totalSs += d * d; }
            double std = Math.Sqrt(totalSs / n);

            double[] classSum = new double[classCount];
            int[] classN = new int[classCount];
            for (int i = 0; i < n; i++)
            {
                int c = actuals[i];
                if (c < 0 || c >= classCount) continue;
                classSum[c] += col[i];
                classN[c]++;
            }
            double between = 0.0;
            for (int c = 0; c < classCount; c++)
            {
                if (classN[c] == 0) continue;
                double d = classSum[c] / classN[c] - mean;
                between += classN[c] * d * d;
            }
            double eta2 = totalSs > 1e-12 ? between / totalSs : 0.0;

            double drop = 0.0;
            for (int r = 0; r < repeats; r++)
            {
                double[] shuffled = (double[])col.Clone();
                for (int i = n - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
                }

                for (int i = 0; i < n; i++) encoded[i].Features[idx] = shuffled[i];
                int[] preds = new int[n];
                for (int i = 0; i < n; i++) preds[i] = Predict(encoded[i].Features);
                drop += baseF1 - MacroF1Metrics.Compute(actuals, preds, classCount).MacroF1;
                for (int i = 0; i < n; i++) encoded[i].Features[idx] = col[i];
            }
            drop /= repeats;

            string name = a < KaggleEcgSample.AuxFeatureNames.Length ? KaggleEcgSample.AuxFeatureNames[a] : $"aux_{a}";
            rows[a] = (a, name, mn, mx, mean, std, eta2, drop);
        }

        var ranked = rows.OrderByDescending(r => r.Drop).ThenByDescending(r => r.Eta2).ToArray();

        Console.WriteLine("\n==========================================================================");
        Console.WriteLine("   ECG DIAGNOSTICS: AUXILIARY FEATURE IMPORTANCE (PERMUTATION, dMacroF1)   ");
        Console.WriteLine("==========================================================================");
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "Evaluated on {0:N0} samples | Baseline Macro F1 = {1:F4} | Repeats = {2}", n, baseF1, repeats));
        Console.WriteLine("--------------------------------------------------------------------------");
        Console.WriteLine(" Rank | Idx | Feature             |    Min    |    Max    |   Mean   |   Std    | eta2  | dMacroF1");
        Console.WriteLine("--------------------------------------------------------------------------");

        int rank = 1;
        foreach (var row in ranked)
        {
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                " {0,4} | {1,3} | {2,-19} | {3,9:F4} | {4,9:F4} | {5,8:F4} | {6,8:F4} | {7,5:F3} | {8,8:F5}",
                rank++, row.Index, row.Name, row.Min, row.Max, row.Mean, row.Std, row.Eta2, row.Drop));
        }

        Console.WriteLine("--------------------------------------------------------------------------");
        int dead = rows.Count(r => r.Std < 1e-6);
        int negligible = rows.Count(r => r.Drop < 0.0005);
        Console.WriteLine($" Dead/constant features (std ~ 0): {dead} | Negligible impact (dMacroF1 < 0.0005): {negligible} / {auxCount}");
        Console.WriteLine(" eta2 = between-class variance / total variance [0..1]. dMacroF1 = Macro F1 drop when the column is shuffled.");
        Console.WriteLine(" dMacroF1 ~ 0 -> little unique signal (removal candidate); std ~ 0 -> dead feature.");
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
