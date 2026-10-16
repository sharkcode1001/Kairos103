using System.Globalization;

namespace Kairos103.Core.Data;

/// <summary>
/// Structured container for an ECG sample from the Kaggle competition.
/// Separates the raw waveform from auxiliary cardiac rhythm features (RR intervals) and metadata.
/// </summary>
public sealed record KaggleEcgSample(
    string Id,
    double[] Waveform,
    double PreRr,
    double PostRr,
    double RrRatio,
    int Label)
{
    public const int WaveformLength = 250;
    public const int AuxCount = 18;
    public const int TotalLength = WaveformLength + AuxCount;

    /// <summary>
    /// Combines the normalized 250 waveform samples followed by 18 patient-independent relative clinical features into a single continuous vector:
    /// [0..2]: pre_rr, post_rr, rr_ratio (cardiac rhythm)
    /// [3..8]: prematurity_index, compensatory_pause, rel_qrs_energy, rel_qrs_p2p, rel_qrs_width, qrs_asymmetry (relative morphology)
    /// [9..17]: norm_baseline, qrs_polarity, t_polarity, stt_discordance, max_pos_slope, max_neg_slope, slope_ratio, rel_p_energy, t_to_qrs_ratio (clinical electrophysiology)
    /// </summary>
    public double[] ToCombinedSignal()
    {
        double[] combined = new double[TotalLength];
        int waveLen = Math.Min(Waveform.Length, WaveformLength);

        // 1. Per-beat min-max waveform normalization: makes LIF spike dynamics strictly patient-independent
        double maxVal = double.NegativeInfinity, minVal = double.PositiveInfinity;
        double totalSq = 0.0;
        for (int i = 0; i < waveLen; i++)
        {
            double v = Waveform[i];
            if (v > maxVal) maxVal = v;
            if (v < minVal) minVal = v;
            totalSq += v * v;
        }

        double p2p = (maxVal > minVal) ? maxVal - minVal : 0.0;
        double invP2p = p2p > 1e-6 ? 1.0 / p2p : 1.0;
        double offset = p2p > 1e-6 ? minVal : 0.0;

        for (int i = 0; i < waveLen; i++)
        {
            combined[i] = (Waveform[i] - offset) * invP2p;
        }

        // Isoelectric baseline reference (mean of beat outer boundaries [0..15] and [234..249])
        double baselineSum = 0.0;
        int baselineCount = 0;
        for (int i = 0; i < Math.Min(16, waveLen); i++)
        {
            baselineSum += Waveform[i];
            baselineCount++;
        }
        for (int i = Math.Max(0, waveLen - 16); i < waveLen; i++)
        {
            baselineSum += Waveform[i];
            baselineCount++;
        }
        double isoelectricBaseline = baselineCount > 0 ? baselineSum / baselineCount : 0.0;

        // Aux 0..2: Rhythm features
        combined[WaveformLength] = PreRr;
        combined[WaveformLength + 1] = PostRr;
        combined[WaveformLength + 2] = RrRatio;

        // Aux 3..8: Dimensionless relative morphology and rhythm metrics (invariant to patient anatomy)
        double totalEnergy = waveLen > 0 ? totalSq / waveLen : 0.0;

        // QRS zone: samples 85 to 165 (ventricular depolarization window protected from edge noise)
        int qrsStart = Math.Min(85, Math.Max(0, waveLen - 1));
        int qrsEnd = Math.Min(165, waveLen);
        double qrsMax = double.NegativeInfinity, qrsMin = double.PositiveInfinity;
        double qrsSq = 0.0;
        double qrsAbsMax = 0.0;
        int peakIdx = qrsStart;
        double qrsNetSum = 0.0;

        for (int i = qrsStart; i < qrsEnd; i++)
        {
            double v = Waveform[i];
            if (v > qrsMax) qrsMax = v;
            if (v < qrsMin) qrsMin = v;
            qrsSq += v * v;
            qrsNetSum += (v - isoelectricBaseline);
            double absV = Math.Abs(v);
            if (absV > qrsAbsMax)
            {
                qrsAbsMax = absV;
                peakIdx = i; // Accurately captures depolarization peak for both positive R and negative QS waves
            }
        }

        double qrsP2p = (qrsMax > qrsMin) ? qrsMax - qrsMin : 0.0;
        double qrsEnergy = (qrsEnd > qrsStart) ? qrsSq / (qrsEnd - qrsStart) : 0.0;

        int qrsWidthCount = 0;
        double halfPeak = 0.5 * qrsAbsMax;
        if (halfPeak > 1e-4)
        {
            for (int i = qrsStart; i < qrsEnd; i++)
            {
                if (Math.Abs(Waveform[i]) >= halfPeak) qrsWidthCount++;
            }
        }

        // T-wave zone: samples 165 to 225 (ventricular repolarization window)
        int tStart = Math.Min(165, Math.Max(0, waveLen - 1));
        int tEnd = Math.Min(225, waveLen);
        double tNetSum = 0.0;
        double tMax = double.NegativeInfinity, tMin = double.PositiveInfinity;
        for (int i = tStart; i < tEnd; i++)
        {
            double v = Waveform[i];
            tNetSum += (v - isoelectricBaseline);
            if (v > tMax) tMax = v;
            if (v < tMin) tMin = v;
        }
        double tP2p = (tMax > tMin) ? tMax - tMin : 0.0;

        // P-wave zone: samples 25 to 85 (atrial depolarization window)
        int pStart = Math.Min(25, Math.Max(0, waveLen - 1));
        int pEnd = Math.Min(85, waveLen);
        double pSq = 0.0;
        for (int i = pStart; i < pEnd; i++)
        {
            double diff = Waveform[i] - isoelectricBaseline;
            pSq += diff * diff;
        }
        double pEnergy = (pEnd > pStart) ? pSq / (pEnd - pStart) : 0.0;

        // QRS slopes
        double maxPosSlope = 0.0;
        double maxNegSlope = 0.0;
        for (int i = qrsStart + 1; i < qrsEnd - 1; i++)
        {
            double slope = (Waveform[i + 1] - Waveform[i - 1]) / (2.0 * Math.Max(1e-4, p2p));
            if (slope > maxPosSlope) maxPosSlope = slope;
            if (-slope > maxNegSlope) maxNegSlope = -slope;
        }

        // Relative features bounded in [0, 1]
        double prematurityIndex = (PreRr + PostRr > 1e-4)
            ? 0.5 * (Math.Clamp((PreRr - PostRr) / (PreRr + PostRr), -1.0, 1.0) + 1.0)
            : 0.5;
        double compensatoryPause = 0.5 * (Math.Clamp((PostRr - PreRr) / Math.Max(1e-4, PreRr) / 2.0, -1.0, 1.0) + 1.0);
        double relQrsEnergy = totalEnergy > 1e-6 ? Math.Clamp(qrsEnergy / totalEnergy, 0.0, 1.0) : 0.0;
        double relQrsP2p = p2p > 1e-6 ? Math.Clamp(qrsP2p / p2p, 0.0, 1.0) : 0.0;
        double relQrsWidth = Math.Clamp(qrsWidthCount / 25.0, 0.0, 1.0);
        double qrsAsymmetry = (qrsEnd > qrsStart) ? (double)(peakIdx - qrsStart) / (qrsEnd - qrsStart) : 0.5;

        // Clinical electrophysiology features [9..17]
        double normBaseline = Math.Clamp((isoelectricBaseline - offset) * invP2p, 0.0, 1.0);
        double qrsNetMean = (qrsEnd > qrsStart) ? qrsNetSum / ((qrsEnd - qrsStart) * Math.Max(1e-4, p2p)) : 0.0;
        double qrsPolarity = Math.Clamp(0.5 * ((qrsNetMean * 3.0) + 1.0), 0.0, 1.0);

        double tNetMean = (tEnd > tStart) ? tNetSum / ((tEnd - tStart) * Math.Max(1e-4, p2p)) : 0.0;
        double tPolarity = Math.Clamp(0.5 * ((tNetMean * 3.0) + 1.0), 0.0, 1.0);

        // ST-T Discordance: discordant repolarization (opposite signs) yields high value > 0.5
        double sttDiscordance = Math.Clamp(0.5 * (1.0 - (qrsNetMean * tNetMean * 20.0)), 0.0, 1.0);

        double normMaxPosSlope = Math.Clamp(maxPosSlope * 4.0, 0.0, 1.0);
        double normMaxNegSlope = Math.Clamp(maxNegSlope * 4.0, 0.0, 1.0);
        double slopeRatio = (maxPosSlope + maxNegSlope > 1e-6) ? maxPosSlope / (maxPosSlope + maxNegSlope) : 0.5;
        double relPwaveEnergy = totalEnergy > 1e-6 ? Math.Clamp(pEnergy / totalEnergy * 5.0, 0.0, 1.0) : 0.0;
        double tToQrsRatio = Math.Clamp(tP2p / Math.Max(1e-4, qrsP2p), 0.0, 1.0);

        combined[WaveformLength + 3] = prematurityIndex;
        combined[WaveformLength + 4] = compensatoryPause;
        combined[WaveformLength + 5] = relQrsEnergy;
        combined[WaveformLength + 6] = relQrsP2p;
        combined[WaveformLength + 7] = relQrsWidth;
        combined[WaveformLength + 8] = qrsAsymmetry;
        combined[WaveformLength + 9] = normBaseline;
        combined[WaveformLength + 10] = qrsPolarity;
        combined[WaveformLength + 11] = tPolarity;
        combined[WaveformLength + 12] = sttDiscordance;
        combined[WaveformLength + 13] = normMaxPosSlope;
        combined[WaveformLength + 14] = normMaxNegSlope;
        combined[WaveformLength + 15] = slopeRatio;
        combined[WaveformLength + 16] = relPwaveEnergy;
        combined[WaveformLength + 17] = tToQrsRatio;

        return combined;
    }

    /// <summary>
    /// Converts this sample into a Kairos SignalSample with effective length set to 250.
    /// The LIF front-end processes only the 250 waveform points, and the 9 aux features are fused into the readout.
    /// </summary>
    public SignalSample ToSignalSample()
    {
        return new SignalSample(Label, ToCombinedSignal(), EffectiveLength: WaveformLength);
    }
}

/// <summary>
/// Specialized loader and preprocessor for the Kaggle ECG Heartbeat Arrhythmia Classification competition dataset.
/// </summary>
public static class KaggleEcgDataset
{
    /// <summary>
    /// Loads the Kaggle train.csv file with automatic column resolution, label verification, and NaN sanitization.
    /// </summary>
    public static List<KaggleEcgSample> LoadTrain(string csvPath)
    {
        return LoadCsv(csvPath, isTest: false);
    }

    /// <summary>
    /// Loads the Kaggle test.csv file, preserving exact sample order with dummy label -1.
    /// </summary>
    public static List<KaggleEcgSample> LoadTest(string csvPath)
    {
        return LoadCsv(csvPath, isTest: true);
    }

    /// <summary>
    /// Reads sample_submission.csv and returns the list of expected ID strings in their exact original order.
    /// </summary>
    public static List<string> LoadSampleSubmissionIds(string csvPath)
    {
        if (!File.Exists(csvPath))
            throw new FileNotFoundException($"Sample submission file not found at: {csvPath}");

        var ids = new List<string>();
        using var reader = new StreamReader(csvPath);
        string? header = reader.ReadLine();
        if (header == null) return ids;

        var headerCols = header.Split(',').Select(c => c.Trim()).ToArray();
        int idIdx = Array.FindIndex(headerCols, c => c.Equals("id", StringComparison.OrdinalIgnoreCase));
        if (idIdx < 0) idIdx = 0;

        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(',');
            if (idIdx < parts.Length)
            {
                ids.Add(parts[idIdx].Trim());
            }
        }

        return ids;
    }

    /// <summary>
    /// Performs stratified train/validation split ensuring exact class proportions across all 4 classes.
    /// </summary>
    public static (List<KaggleEcgSample> Train, List<KaggleEcgSample> Val) StratifiedSplit(
        IReadOnlyList<KaggleEcgSample> samples,
        double valFraction = 0.2,
        int seed = 42)
    {
        if (valFraction <= 0.0)
        {
            var all = samples.ToList();
            Shuffle(all, new Random(seed));
            return (all, new List<KaggleEcgSample>());
        }

        if (valFraction >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(valFraction), "Validation fraction must be between 0 and 1.");

        var groups = samples.GroupBy(s => s.Label).ToDictionary(g => g.Key, g => g.ToList());
        var trainList = new List<KaggleEcgSample>();
        var valList = new List<KaggleEcgSample>();

        var rng = new Random(seed);

        foreach (var kvp in groups)
        {
            var list = kvp.Value;
            // Fisher-Yates shuffle
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }

            int valCount = (int)Math.Round(list.Count * valFraction);
            valCount = Math.Clamp(valCount, 1, list.Count - 1);

            valList.AddRange(list.Take(valCount));
            trainList.AddRange(list.Skip(valCount));
        }

        // Shuffle resulting sets for uniform presentation during epoch training
        Shuffle(trainList, rng);
        Shuffle(valList, rng);

        return (trainList, valList);
    }

    /// <summary>
    /// Partitions samples into K stratified folds preserving class distribution across each fold.
    /// </summary>
    public static List<List<KaggleEcgSample>> StratifiedKFolds(IReadOnlyList<KaggleEcgSample> samples, int kFolds = 5, int seed = 42)
    {
        if (kFolds < 2) throw new ArgumentOutOfRangeException(nameof(kFolds), "kFolds must be >= 2");
        var folds = new List<List<KaggleEcgSample>>(kFolds);
        for (int k = 0; k < kFolds; k++) folds.Add(new List<KaggleEcgSample>());

        var groups = samples.GroupBy(s => s.Label).ToDictionary(g => g.Key, g => g.ToList());
        var rng = new Random(seed);

        foreach (var kvp in groups)
        {
            var list = kvp.Value;
            Shuffle(list, rng);
            for (int i = 0; i < list.Count; i++)
            {
                folds[i % kFolds].Add(list[i]);
            }
        }

        for (int k = 0; k < kFolds; k++)
        {
            Shuffle(folds[k], rng);
        }

        return folds;
    }

    /// <summary>
    /// Computes balanced class weights inversely proportional to class frequencies:
    /// w_c = (N / (C * N_c))^power
    /// power=1.0 is exact inverse frequency; power=0.5 (default) is smoothed square-root inverse frequency to prevent rare class over-prediction.
    /// </summary>
    public static double[] ComputeBalancedClassWeights(IReadOnlyList<KaggleEcgSample> samples, int classCount = 4, double power = 0.5)
    {
        int[] counts = new int[classCount];
        int total = 0;
        foreach (var s in samples)
        {
            if (s.Label >= 0 && s.Label < classCount)
            {
                counts[s.Label]++;
                total++;
            }
        }

        double[] weights = new double[classCount];
        for (int c = 0; c < classCount; c++)
        {
            if (counts[c] > 0)
            {
                double raw = (double)total / (classCount * counts[c]);
                weights[c] = Math.Pow(raw, power);
            }
            else
            {
                weights[c] = 1.0;
            }
        }

        return weights;
    }

    private static void Shuffle<T>(List<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static List<KaggleEcgSample> LoadCsv(string csvPath, bool isTest)
    {
        if (!File.Exists(csvPath))
            throw new FileNotFoundException($"File not found at: {csvPath}");

        using var reader = new StreamReader(csvPath);
        string? headerLine = reader.ReadLine();
        if (headerLine == null)
            throw new InvalidDataException("CSV file is completely empty.");

        var headers = headerLine.Split(',').Select(h => h.Trim()).ToArray();

        int idCol = -1;
        int labelCol = -1;
        int preRrCol = -1;
        int postRrCol = -1;
        int rrRatioCol = -1;

        var sigCols = new List<int>();

        for (int i = 0; i < headers.Length; i++)
        {
            string col = headers[i].ToLowerInvariant();
            if (col == "id") idCol = i;
            else if (col == "label" || col == "target") labelCol = i;
            else if (col == "pre_rr") preRrCol = i;
            else if (col == "post_rr") postRrCol = i;
            else if (col == "rr_ratio") rrRatioCol = i;
            else if (col.StartsWith("sig_")) sigCols.Add(i);
        }

        // Fallback for signal columns if no "sig_" prefix exists
        if (sigCols.Count == 0)
        {
            for (int i = 0; i < headers.Length; i++)
            {
                if (i != idCol && i != labelCol && i != preRrCol && i != postRrCol && i != rrRatioCol)
                {
                    sigCols.Add(i);
                }
            }
        }

        if (!isTest && labelCol < 0)
            throw new InvalidDataException("Train dataset must contain a 'label' or 'target' column.");

        var samples = new List<KaggleEcgSample>();
        string? line;
        int rowIdx = 1;

        while ((line = reader.ReadLine()) != null)
        {
            rowIdx++;
            if (string.IsNullOrWhiteSpace(line)) continue;

            var tokens = line.Split(',');

            string id = idCol >= 0 && idCol < tokens.Length ? tokens[idCol].Trim() : (rowIdx - 2).ToString();

            int label = -1;
            if (!isTest)
            {
                if (labelCol >= tokens.Length || !int.TryParse(tokens[labelCol].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out label))
                {
                    throw new InvalidOperationException($"Row {rowIdx}: Missing or invalid label value '{tokens.ElementAtOrDefault(labelCol)}'.");
                }

                if (label < 0 || label > 3)
                {
                    throw new InvalidOperationException($"Row {rowIdx}: Label {label} is out of bounds. Expected 0, 1, 2, or 3.");
                }
            }

            double preRr = 0.0, postRr = 0.0, rrRatio = 1.0;
            if (preRrCol >= 0 && preRrCol < tokens.Length)
                preRr = ParseSafeDouble(tokens[preRrCol], 0.0);
            if (postRrCol >= 0 && postRrCol < tokens.Length)
                postRr = ParseSafeDouble(tokens[postRrCol], 0.0);
            if (rrRatioCol >= 0 && rrRatioCol < tokens.Length)
                rrRatio = ParseSafeDouble(tokens[rrRatioCol], 1.0);

            double[] wave = new double[KaggleEcgSample.WaveformLength];
            int takeCount = Math.Min(sigCols.Count, KaggleEcgSample.WaveformLength);
            for (int k = 0; k < takeCount; k++)
            {
                int colIdx = sigCols[k];
                if (colIdx < tokens.Length)
                {
                    wave[k] = ParseSafeDouble(tokens[colIdx], 0.0);
                }
            }

            samples.Add(new KaggleEcgSample(id, wave, preRr, postRr, rrRatio, label));
        }

        return samples;
    }

    private static double ParseSafeDouble(string token, double fallback)
    {
        string t = token.Trim();
        if (string.IsNullOrEmpty(t) || t.Equals("nan", StringComparison.OrdinalIgnoreCase) || t.Equals("null", StringComparison.OrdinalIgnoreCase))
        {
            return fallback;
        }

        if (double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out double val))
        {
            if (double.IsNaN(val) || double.IsInfinity(val)) return fallback;
            return val;
        }

        return fallback;
    }
}
