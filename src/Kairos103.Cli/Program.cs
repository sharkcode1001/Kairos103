using Kairos103.Cli.Commands;

Console.WriteLine("====================================================================");
Console.WriteLine("   Kairos103: Clockless Rank-Order Neural Architecture (.NET 10)    ");
Console.WriteLine("====================================================================\n");

string command = args.Length > 0 ? args[0].ToLowerInvariant() : "benchmark";

switch (command)
{
    case "benchmark":
    {
        string dataset = "synthetic";
        string dir = "datasets";

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--dataset" && i + 1 < args.Length)
            {
                dataset = args[++i];
            }
            else if (args[i] == "--datasets-dir" && i + 1 < args.Length)
            {
                dir = args[++i];
            }
        }

        RunBenchmarkCommand.Execute(datasetsDir: dir, targetDataset: dataset);
        break;
    }

    case "jitter-sweep":
    {
        SweepNoiseCommand.Execute();
        break;
    }

    case "export-plots":
    case "export-report":
    {
        string output = "./results";
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--output" && i + 1 < args.Length)
            {
                output = args[++i];
            }
        }
        ExportReportCommand.Execute(outputPath: output);
        break;
    }

    default:
    {
        Console.WriteLine($"Comando desconocido: {command}");
        Console.WriteLine("Uso disponible:");
        Console.WriteLine("  dotnet run --project src/Kairos103.Cli -- benchmark [--dataset synthetic|ecg] [--datasets-dir <path>]");
        Console.WriteLine("  dotnet run --project src/Kairos103.Cli -- jitter-sweep");
        Console.WriteLine("  dotnet run --project src/Kairos103.Cli -- export-plots [--output <path>]");
        break;
    }
}
