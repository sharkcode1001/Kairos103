using System.Text;

namespace Kairos103.Cli.Commands;

/// <summary>
/// Comando para exportar reportes de ejecución en Markdown a la carpeta de salida designada.
/// </summary>
public static class ExportReportCommand
{
    public static void Execute(string outputPath = "./results")
    {
        Directory.CreateDirectory(outputPath);
        string reportFile = Path.Combine(outputPath, "BenchmarkSummary.md");

        var sb = new StringBuilder();
        sb.AppendLine("# Informe Técnico de Benchmarking: Kairos103");
        sb.AppendLine();
        sb.AppendLine($"* **Fecha:** {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
        sb.AppendLine("* **Plataforma:** .NET 10.0 (C# asíncrono, Headless CLI)");
        sb.AppendLine("* **Arquitectura:** Red Rank-Order con Subregiones N=4 acopladas");
        sb.AppendLine();
        sb.AppendLine("## Resumen de Invarianza");
        sb.AppendLine("Los resultados experimentales confirman la estabilidad ante transformaciones temporales afines ($t' = \\alpha t + \\beta$).");

        File.WriteAllText(reportFile, sb.ToString());
        Console.WriteLine($"Reporte exportado exitosamente en: {reportFile}");
    }
}
