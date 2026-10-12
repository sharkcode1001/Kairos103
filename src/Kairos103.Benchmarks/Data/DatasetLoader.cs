using System.Globalization;

namespace Kairos103.Benchmarks.Data;

/// <summary>
/// Record representing a time series sample with variable effective length and zero padding.
/// </summary>
public sealed record TimeSeriesSample(int Label, int Length, double[] Values);

/// <summary>
/// CSV dataset reader and loader designed for the Kairos103 benchmark protocol.
/// </summary>
public static class DatasetLoader
{
    public static List<TimeSeriesSample> LoadCsv(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Dataset file not found: {filePath}", filePath);

        var samples = new List<TimeSeriesSample>();
        using var reader = new StreamReader(filePath);
        string? header = reader.ReadLine(); // Skip header row

        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            string[] parts = line.Split(',');
            int label = int.Parse(parts[0], CultureInfo.InvariantCulture);
            int length = int.Parse(parts[1], CultureInfo.InvariantCulture);

            double[] values = new double[parts.Length - 2];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = double.Parse(parts[i + 2], CultureInfo.InvariantCulture);
            }

            samples.Add(new TimeSeriesSample(label, length, values));
        }

        return samples;
    }
}
