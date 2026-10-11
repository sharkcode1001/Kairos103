using System.Globalization;

namespace Kairos103.Benchmarks.Data;

/// <summary>
/// Registro que representa una muestra de serie temporal con longitud variable y ceros de relleno.
/// </summary>
public sealed record TimeSeriesSample(int Label, int Length, double[] Values);

/// <summary>
/// Lector y cargador de datasets CSV organizados para el protocolo de Kairos103.
/// </summary>
public static class DatasetLoader
{
    public static List<TimeSeriesSample> LoadCsv(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"No se encontró el archivo de dataset: {filePath}", filePath);

        var samples = new List<TimeSeriesSample>();
        using var reader = new StreamReader(filePath);
        string? header = reader.ReadLine(); // Saltar encabezado

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
