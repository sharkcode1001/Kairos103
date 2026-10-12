namespace Kairos103.Cli.Localization;

/// <summary>
/// Provides internationalization support for CLI console messages.
/// Default language is English ("en"), with full support for Spanish ("es").
/// </summary>
public static class CliStrings
{
    public static string CurrentLanguage { get; set; } = "en";

    public static bool IsSpanish => CurrentLanguage.Equals("es", StringComparison.OrdinalIgnoreCase);

    public static string BannerTitle =>
        IsSpanish
            ? "=== PROTOCOLO DE BENCHMARK KAIROS103 ==="
            : "=== KAIROS103 BENCHMARK PROTOCOL ===";

    public static string DatasetAndDir(string dataset, string dir) =>
        IsSpanish
            ? $"Dataset: {dataset} | Directorio: {dir}"
            : $"Dataset: {dataset} | Directory: {dir}";

    public static string FastModeNotice =>
        IsSpanish
            ? "Modo: RÁPIDO (--fast) activado para verificación ágil."
            : "Mode: FAST (--fast) enabled for quick verification.";

    public static string DatasetNotFound(string path, string currentDir) =>
        IsSpanish
            ? $"[ERROR] No se encontró el dataset de entrenamiento en: {path}\n        Ruta de búsqueda actual: {currentDir}"
            : $"[ERROR] Training dataset not found at: {path}\n        Current search path: {currentDir}";

    public static string Step1LoadingTrain =>
        IsSpanish
            ? "[1/4] Cargando dataset de entrenamiento nominal (1.0x)..."
            : "[1/4] Loading nominal training dataset (1.0x)...";

    public static string SamplesLoaded(int samples, int classes) =>
        IsSpanish
            ? $"      Muestras cargadas: {samples} | Clases: {classes}"
            : $"      Samples loaded: {samples} | Classes: {classes}";

    public static string Step2TrainingModels =>
        IsSpanish
            ? "\n[2/4] Entrenando Modelos en Velocidad Nominal (1.0x)..."
            : "\n[2/4] Training Models at Nominal Speed (1.0x)...";

    public static string TrainingKairos =>
        IsSpanish
            ? "      [1/3] Entrenando Kairos103 (Rank-Order Modular N=4)... "
            : "      [1/3] Training Kairos103 (Rank-Order Modular N=4)... ";

    public static string TrainingTransformer =>
        IsSpanish
            ? "      [2/3] Entrenando Mini-Transformer (Attention + Positional Encodings)... "
            : "      [2/3] Training Mini-Transformer (Attention + Positional Encodings)... ";

    public static string TrainingLstm =>
        IsSpanish
            ? "      [3/3] Entrenando Simple LSTM (Estado Recurrente)... "
            : "      [3/3] Training Simple LSTM (Recurrent Hidden State)... ";

    public static string Done =>
        IsSpanish ? "LISTO" : "DONE";

    public static string Step3EvaluatingWarp =>
        IsSpanish
            ? "\n[3/4] Evaluando Invarianza a Time-Warping en Inferencia..."
            : "\n[3/4] Evaluating Time-Warp Invariance during Inference...";

    public static string EvaluatingFactor(double factor) =>
        IsSpanish
            ? $"  Evaluando factor {factor:F1}x ..."
            : $"  Evaluating factor {factor:F1}x ...";

    public static string Step4Summary =>
        IsSpanish
            ? "\n[4/4] Resumen de Resultados y Métricas"
            : "\n[4/4] Results Summary and Metrics";

    public static string EfficiencyMetricsHeader =>
        IsSpanish
            ? "\n--- Métricas de Eficiencia Computacional ---"
            : "\n--- Computational Efficiency Metrics ---";

    public static string FrontEndMacs(long macs) =>
        IsSpanish
            ? $"Kairos Front-End MACs (continuos):    {macs:N0}"
            : $"Kairos Front-End MACs (continuous):   {macs:N0}";

    public static string Spikes(long spikes) =>
        IsSpanish
            ? $"Kairos Spikes Discretos (eventos):    {spikes:N0}"
            : $"Kairos Discrete Spikes (events):       {spikes:N0}";

    public static string ReadoutOps(long ops) =>
        IsSpanish
            ? $"Kairos Acumulaciones (clasificador):  {ops:N0}"
            : $"Kairos Readout Accumulations:         {ops:N0}";

    public static string TransformerMacs(long macs) =>
        $"Mini-Transformer MACs:                {macs:N0}";

    public static string LstmMacs(long macs) =>
        $"Simple LSTM MACs:                     {macs:N0}";

    public static string HeapMemory(long bytes) =>
        IsSpanish
            ? $"Memoria Heap Asignada:                {bytes:N0} bytes"
            : $"Allocated Heap Memory:                {bytes:N0} bytes";

    public static string JitterSweepHeader(double maxJitter, double step) =>
        IsSpanish
            ? $"=== BARRIDO DE ESTABILIDAD A JITTER KAIROS103 ===\nJitter Máximo: {maxJitter:F3} | Paso: {step:F3}\n"
            : $"=== KAIROS103 JITTER STABILITY SWEEP ===\nMax Jitter: {maxJitter:F3} | Step: {step:F3}\n";

    public static string JitterTableHeader =>
        IsSpanish
            ? "Jitter (stdDev) | Precisión Kairos103\n----------------|--------------------"
            : "Jitter (stdDev) | Kairos103 Accuracy\n----------------|--------------------";

    public static string ExportStarting(string dataset) =>
        IsSpanish
            ? $"\n--- Exportando resultados para dataset: {dataset} ---"
            : $"\n--- Exporting results for dataset: {dataset} ---";

    public static string SvgGenerated(string path) =>
        IsSpanish
            ? $"[OK] Gráfico vectorial generado: {path}"
            : $"[OK] Vector plot generated: {path}";

    public static string ReportGenerated(string path) =>
        IsSpanish
            ? $"\n[OK] Informe consolidado generado en: {path}"
            : $"\n[OK] Consolidated report generated at: {path}";

    public static string UnknownCommand(string command) =>
        IsSpanish
            ? $"Comando desconocido: {command}"
            : $"Unknown command: {command}";

    public static string AvailableUsage =>
        IsSpanish
            ? "Uso disponible:\n" +
              "  dotnet run --project src/Kairos103.Cli -- benchmark [--dataset synthetic|ecg] [--warp-factors 0.5,0.8,1.0,1.2,2.0] [--runs 1] [--fast] [--lang en|es]\n" +
              "  dotnet run --project src/Kairos103.Cli -- jitter-sweep [--max-jitter-ms 50] [--step 10] [--lang en|es]\n" +
              "  dotnet run --project src/Kairos103.Cli -- export-plots [--output <path>] [--dataset synthetic|ecg] [--full] [--lang en|es]"
            : "Available usage:\n" +
              "  dotnet run --project src/Kairos103.Cli -- benchmark [--dataset synthetic|ecg] [--warp-factors 0.5,0.8,1.0,1.2,2.0] [--runs 1] [--fast] [--lang en|es]\n" +
              "  dotnet run --project src/Kairos103.Cli -- jitter-sweep [--max-jitter-ms 50] [--step 10] [--lang en|es]\n" +
              "  dotnet run --project src/Kairos103.Cli -- export-plots [--output <path>] [--dataset synthetic|ecg] [--full] [--lang en|es]";
}
