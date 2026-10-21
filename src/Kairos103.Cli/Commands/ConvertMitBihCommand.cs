using Kairos103.Core.Data;

namespace Kairos103.Cli.Commands;

public static class ConvertMitBihCommand
{
    public static void Execute(string input, string output, string audit)
    {
        Console.WriteLine("Convirtiendo MIT-BIH WFDB a CSV compatible con NPPE2...");
        var result = MitBihDatasetConverter.Convert(input, output, audit);
        Console.WriteLine($"Registros: {result.Records}");
        Console.WriteLine($"Filas generadas: {result.Rows:N0}");
        Console.WriteLine($"Clases: {string.Join(", ", result.Labels.OrderBy(x => x.Key).Select(x => $"{x.Key}={x.Value:N0}"))}");
        Console.WriteLine($"CSV: {output}");
        Console.WriteLine($"Auditoría: {audit}");
    }
}
