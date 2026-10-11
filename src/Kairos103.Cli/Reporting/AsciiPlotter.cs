using System.Text;

namespace Kairos103.Cli.Reporting;

/// <summary>
/// Generador de gráficos ASCII para terminal y exportación a tablas Markdown.
/// </summary>
public static class AsciiPlotter
{
    /// <summary>
    /// Renderiza un gráfico de barras comparativo en la consola.
    /// </summary>
    public static void RenderBar(string label, double value, double maxValue = 1.0, int width = 30)
    {
        int filled = (int)Math.Round((Math.Clamp(value, 0.0, maxValue) / maxValue) * width);
        string bar = new string('█', filled) + new string('░', width - filled);
        Console.WriteLine($"{label,-15} |{bar}| {value * 100.0,6:F2}%");
    }

    /// <summary>
    /// Genera una tabla comparativa en formato Markdown.
    /// </summary>
    public static string GenerateMarkdownTable(
        string datasetName,
        double[] factors,
        double[] kairosAcc,
        double[] transformerAcc,
        double[] lstmAcc)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"### Resultados de Benchmark: {datasetName}");
        sb.AppendLine();
        sb.AppendLine("| Factor Warp | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |");
        sb.AppendLine("|:-----------:|:----------------:|:----------------:|:-----------:|");

        for (int i = 0; i < factors.Length; i++)
        {
            sb.AppendLine($"| {factors[i]:F1}x | {kairosAcc[i] * 100.0:F2}% | {transformerAcc[i] * 100.0:F2}% | {lstmAcc[i] * 100.0:F2}% |");
        }

        sb.AppendLine();
        return sb.ToString();
    }
}
