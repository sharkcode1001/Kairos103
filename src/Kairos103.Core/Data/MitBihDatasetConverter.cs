using System.Globalization;
using System.Text;

namespace Kairos103.Core.Data;

/// <summary>Reads the common MIT-BIH Arrhythmia WFDB layout and exports NPPE2 rows.</summary>
public static class MitBihDatasetConverter
{
    private static readonly Dictionary<char, int> LabelMap = new()
    {
        ['N'] = 0, ['L'] = 0, ['R'] = 0, ['e'] = 0, ['j'] = 0,
        ['A'] = 1, ['a'] = 1, ['J'] = 1, ['S'] = 1,
        ['V'] = 2, ['E'] = 2,
        ['F'] = 3
    };

    private static readonly Dictionary<int, char> AnnotationSymbols = new()
    {
        [1] = 'N', [2] = 'L', [3] = 'R', [4] = 'a', [5] = 'V', [6] = 'F',
        [7] = 'J', [8] = 'A', [9] = 'S', [10] = 'E', [11] = 'j', [12] = '/',
        [13] = 'Q'
    };

    public sealed record Audit(int Records, int Rows, Dictionary<int, int> Labels, Dictionary<string, int> Skipped);

    public static Audit Convert(string inputDir, string outputCsv, string auditPath, int left = 100, int right = 150)
    {
        if (left + right != KaggleEcgSample.WaveformLength)
            throw new ArgumentException("MIT-BIH window must contain exactly 250 samples.");
        var labels = new Dictionary<int, int>();
        var skipped = new Dictionary<string, int>();
        var rows = new List<string>();
        string header = "id," + string.Join(',', Enumerable.Range(0, 250).Select(i => $"sig_{i}")) + ",pre_rr,post_rr,rr_ratio,label";
        rows.Add(header);
        int records = 0;

        foreach (string hea in Directory.EnumerateFiles(inputDir, "*.hea").OrderBy(x => x))
        {
            string record = Path.Combine(Path.GetDirectoryName(hea)!, Path.GetFileNameWithoutExtension(hea));
            try
            {
                var info = ReadHeader(hea);
                double[] signal = Read212(Path.ChangeExtension(record, ".dat"), info.Samples, info.Gain, info.Baseline);
                var annotations = ReadAnnotations(Path.ChangeExtension(record, ".atr"));
                var kept = annotations.Where(a => LabelMap.ContainsKey(a.Symbol)).ToList();
                records++;
                for (int i = 1; i < kept.Count - 1; i++)
                {
                    int pos = kept[i].Sample;
                    int start = pos - left, end = pos + right;
                    if (start < 0 || end > signal.Length) { Inc(skipped, "edge_window"); continue; }
                    double pre = pos - kept[i - 1].Sample, post = kept[i + 1].Sample - pos;
                    if (pre <= 0 || post <= 0) { Inc(skipped, "invalid_rr"); continue; }
                    double[] beat = signal[start..end];
                    double min = beat.Min(), max = beat.Max(), span = max - min;
                    if (!double.IsFinite(span)) { Inc(skipped, "invalid_signal"); continue; }
                    int label = LabelMap[kept[i].Symbol];
                    var sb = new StringBuilder($"mit_{Path.GetFileNameWithoutExtension(record)}_{pos:D7}");
                    foreach (double value in beat) sb.Append(',').Append((span > 1e-12 ? (value - min) / span : 0.0).ToString("R", CultureInfo.InvariantCulture));
                    sb.Append(',').Append(pre.ToString("R", CultureInfo.InvariantCulture));
                    sb.Append(',').Append(post.ToString("R", CultureInfo.InvariantCulture));
                    sb.Append(',').Append((pre / post).ToString("R", CultureInfo.InvariantCulture));
                    sb.Append(',').Append(label);
                    rows.Add(sb.ToString());
                    labels[label] = labels.GetValueOrDefault(label) + 1;
                }
            }
            catch (Exception ex)
            {
                Inc(skipped, ex is FileNotFoundException ? "missing_file" : "read_error");
                Console.WriteLine($"[WARN] {Path.GetFileNameWithoutExtension(record)}: {ex.Message}");
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputCsv))!);
        File.WriteAllLines(outputCsv, rows, System.Text.Encoding.UTF8);
        var audit = new Audit(records, rows.Count - 1, labels, skipped);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(auditPath))!);
        File.WriteAllText(auditPath, System.Text.Json.JsonSerializer.Serialize(audit, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        return audit;
    }

    private sealed record Header(int Samples, double Gain, double Baseline);
    private sealed record Beat(int Sample, char Symbol);

    private static Header ReadHeader(string path)
    {
        string[] lines = File.ReadAllLines(path);
        string[] first = lines[0].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int samples = int.Parse(first[3], CultureInfo.InvariantCulture);
        string[] signal = lines[1].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        double gain = double.Parse(signal[2], CultureInfo.InvariantCulture);
        // WFDB signal line: file, format, gain, adc_res, adc_zero, baseline, ...
        double baseline = double.Parse(signal[5], CultureInfo.InvariantCulture);
        return new Header(samples, gain, baseline);
    }

    private static double[] Read212(string path, int samples, double gain, double baseline)
    {
        byte[] data = File.ReadAllBytes(path);
        var result = new double[samples];
        int output = 0;
        // Each 3-byte group contains one sample from each of two channels.
        // The caller selected MLII as channel 0, so keep only the first
        // 12-bit sample in every group.
        for (int i = 0; i + 2 < data.Length && output < samples; i += 3)
        {
            int a = data[i] | ((data[i + 1] & 0x0F) << 8);
            if ((a & 0x800) != 0) a -= 0x1000;
            result[output++] = (a - baseline) / Math.Max(1e-12, gain);
        }
        if (output != samples) throw new InvalidDataException($"Expected {samples} samples, decoded {output}.");
        return result;
    }

    private static List<Beat> ReadAnnotations(string path)
    {
        byte[] data = File.ReadAllBytes(path);
        var result = new List<Beat>();
        int sample = 0, i = 0;
        while (i + 1 < data.Length)
        {
            int code = (data[i] >> 2) & 0x3F;
            // WFDB annotation word: low 2 bits of the first byte are the
            // high interval bits, followed by the complete second byte.
            int interval = ((data[i] & 0x03) << 8) | data[i + 1];
            i += 2;
            // Some MIT-BIH annotation files contain an internal zero word
            // before the remaining annotation stream. Only treat 0,0 as the
            // terminator when the rest of the file is padding.
            if (code == 0 && interval == 0 && data.Skip(i).All(x => x == 0)) break;
            if (code == 59) { if (i + 3 >= data.Length) break; sample += BitConverter.ToInt32(data, i); i += 4; continue; }
            if (code is 60 or 61 or 62) { if (i + 1 >= data.Length) break; i += 2; continue; }
            if (code == 63) { while (i < data.Length && data[i++] != 0) { } continue; }
            sample += interval;
            if (AnnotationSymbols.TryGetValue(code, out char symbol)) result.Add(new Beat(sample, symbol));
        }
        return result;
    }

    private static void Inc(Dictionary<string, int> dict, string key) => dict[key] = dict.GetValueOrDefault(key) + 1;
}
