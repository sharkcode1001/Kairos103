using Kairos103.Benchmarks.Generators;
using Kairos103.Cli.Localization;
using Kairos103.Cli.Reporting;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

/// <summary>
/// Command to perform progressive temporal jitter sweeps on Kairos103.
/// </summary>
public static class SweepNoiseCommand
{
    public static void Execute(double maxJitter = 0.05, double step = 0.01)
    {
        Console.WriteLine(CliStrings.JitterSweepHeader(maxJitter, step));

        var network = new KairosNetwork(subregionCount: 16);
        var trainer = new PairwiseTrainer(4, network.TotalFeatures);

        // Train on nominal signals
        var trainList = new List<(double[] Features, int Label)>();
        for (int c = 0; c < 4; c++)
        {
            for (int r = 0; r < 50; r++)
            {
                var sig = SyntheticSignals.GenerateSignal(c, 1.0);
                var (feats, _) = network.Forward(sig, sig.Length);
                trainList.Add((feats, c));
            }
        }
        trainer.Train(trainList, epochs: 20);

        Console.WriteLine(CliStrings.JitterTableHeader);

        for (double j = 0.0; j <= maxJitter + 1e-6; j += step)
        {
            var injector = new NoiseInjector(seed: 100);
            int correct = 0;
            int total = 0;

            for (int c = 0; c < 4; c++)
            {
                for (int rep = 0; rep < 40; rep++)
                {
                    var sig = SyntheticSignals.GenerateSignal(c, 1.0);
                    var noisySig = injector.ApplyMonotonicJitter(sig, j, 0.03);
                    var (feats, _) = network.Forward(noisySig, noisySig.Length);
                    var (pred, _) = trainer.Predict(feats);
                    if (pred == c) correct++;
                    total++;
                }
            }

            double acc = (double)correct / total;
            AsciiPlotter.RenderBar($"σ = {j:F3}", acc);
        }
    }
}
