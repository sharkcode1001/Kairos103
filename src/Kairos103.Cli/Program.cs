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
        double[]? factors = null;
        int runs = 1;
        bool fast = false;

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
            else if (args[i] == "--warp-factors" && i + 1 < args.Length)
            {
                string raw = args[++i];
                factors = raw.Split(',', StringSplitOptions.RemoveEmptyEntries)
                             .Select(s => double.Parse(s.Trim(), System.Globalization.CultureInfo.InvariantCulture))
                             .ToArray();
            }
            else if (args[i] == "--runs" && i + 1 < args.Length)
            {
                runs = int.Parse(args[++i]);
            }
            else if (args[i] == "--fast")
            {
                fast = true;
            }
        }

        RunBenchmarkCommand.Execute(datasetsDir: dir, targetDataset: dataset, customFactors: factors, runs: runs, fastMode: fast);
        break;
    }

    case "jitter-sweep":
    {
        double maxJitter = 0.05;
        double step = 0.01;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--max-jitter-ms" && i + 1 < args.Length)
            {
                // Si viene en ms (ej 50), mapear a escala fraccional
                double val = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                maxJitter = val > 1.0 ? val / 1000.0 : val;
            }
            else if (args[i] == "--step" && i + 1 < args.Length)
            {
                double val = double.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                step = val > 1.0 ? val / 1000.0 : val;
            }
        }

        SweepNoiseCommand.Execute(maxJitter: maxJitter, step: step);
        break;
    }

    case "export-plots":
    case "export-report":
    {
        string output = "./results";
        string dataset = "synthetic";
        bool fast = true;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--output" && i + 1 < args.Length)
            {
                output = args[++i];
            }
            else if (args[i] == "--dataset" && i + 1 < args.Length)
            {
                dataset = args[++i];
            }
            else if (args[i] == "--full")
            {
                fast = false;
            }
        }
        ExportReportCommand.Execute(outputPath: output, targetDataset: dataset, fast: fast);
        break;
    }

    default:
    {
        Console.WriteLine($"Comando desconocido: {command}");
        Console.WriteLine("Uso disponible:");
        Console.WriteLine("  dotnet run --project src/Kairos103.Cli -- benchmark [--dataset synthetic|ecg] [--warp-factors 0.5,0.8,1.0,1.2,2.0] [--runs 1] [--fast]");
        Console.WriteLine("  dotnet run --project src/Kairos103.Cli -- jitter-sweep [--max-jitter-ms 50] [--step 10]");
        Console.WriteLine("  dotnet run --project src/Kairos103.Cli -- export-plots [--output <path>] [--dataset synthetic|ecg] [--full]");
        break;
    }
}
