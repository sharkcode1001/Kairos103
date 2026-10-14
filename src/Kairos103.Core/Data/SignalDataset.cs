using System.Globalization;

namespace Kairos103.Core.Data;

/// <summary>
/// Universal representation of a time-series or 1D signal sample.
/// </summary>
public sealed record SignalSample(int Label, double[] Values, int EffectiveLength);

/// <summary>
/// Flexible dataset loader that handles standard ML CSV formats (UCR, MIT-BIH, Kaggle)
/// as well as Kairos103 benchmark formats (label,length,values...).
/// </summary>
public static class SignalDataset
{
    public static List<SignalSample> LoadCsv(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Dataset file not found: {filePath}", filePath);

        var samples = new List<SignalSample>();
        using var reader = new StreamReader(filePath);

        string? firstLine = reader.ReadLine();
        if (firstLine == null) return samples;

        // Check if first line is a header (contains non-numeric characters in first token)
        string[] firstParts = firstLine.Split(',');
        bool hasHeader = !double.TryParse(firstParts[0].Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out _);

        if (!hasHeader)
        {
            samples.Add(ParseLine(firstParts));
        }

        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] parts = line.Split(',');
            samples.Add(ParseLine(parts));
        }

        return samples;
    }

    private static SignalSample ParseLine(string[] parts)
    {
        if (parts.Length < 2)
            throw new FormatException("CSV row must contain at least a label and one signal value.");

        int label = (int)Math.Round(double.Parse(parts[0].Trim(), CultureInfo.InvariantCulture));

        // Detect if parts[1] is an integer length (Kairos format: label, length, v0, v1...)
        // or a continuous float value (standard format: label, v0, v1, v2...)
        bool isKairosFormat = false;
        if (parts.Length > 2 && int.TryParse(parts[1].Trim(), out int candidateLength))
        {
            if (candidateLength > 0 && candidateLength <= parts.Length - 2)
            {
                isKairosFormat = true;
            }
        }

        if (isKairosFormat)
        {
            int length = int.Parse(parts[1].Trim(), CultureInfo.InvariantCulture);
            double[] values = new double[parts.Length - 2];
            for (int i = 0; i < values.Length; i++)
            {
                string p = parts[i + 2].Trim();
                values[i] = double.TryParse(p, NumberStyles.Any, CultureInfo.InvariantCulture, out double v) ? v : 0.0;
            }
            return new SignalSample(label, values, length);
        }
        else
        {
            // Standard format: label, v0, v1, v2...
            double[] values = new double[parts.Length - 1];
            for (int i = 0; i < values.Length; i++)
            {
                string p = parts[i + 1].Trim();
                values[i] = double.TryParse(p, NumberStyles.Any, CultureInfo.InvariantCulture, out double v) ? v : 0.0;
            }
            return new SignalSample(label, values, values.Length);
        }
    }
}
