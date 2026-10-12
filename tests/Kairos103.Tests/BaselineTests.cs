using Kairos103.Benchmarks.Baselines;
using Kairos103.Benchmarks.Data;
using Xunit;

namespace Kairos103.Tests;

public class BaselineTests
{
    [Fact]
    public void MiniTransformer_TrainsAndClassifiesSyntheticPatterns()
    {
        int numClasses = 2;
        int seqLen = 32;
        var transformer = new MiniTransformer(numClasses, dModel: 8, maxSeqLen: seqLen, seed: 123);

        // Generate separable synthetic dataset
        var dataset = new List<TimeSeriesSample>();
        for (int i = 0; i < 40; i++)
        {
            // Class 0: ascending ramp
            double[] v0 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) v0[t] = (double)t / seqLen;
            dataset.Add(new TimeSeriesSample(0, seqLen, v0));

            // Class 1: descending ramp
            double[] v1 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) v1[t] = 1.0 - ((double)t / seqLen);
            dataset.Add(new TimeSeriesSample(1, seqLen, v1));
        }

        // Evaluate before training
        int correctBefore = 0;
        foreach (var sample in dataset)
        {
            var (pred, _) = transformer.Forward(sample.Values, sample.Length);
            if (pred == sample.Label) correctBefore++;
        }

        // Train with Adam backpropagation
        transformer.Train(dataset, epochs: 10, lr: 0.01);

        // Evaluate after training
        int correctAfter = 0;
        foreach (var sample in dataset)
        {
            var (pred, _) = transformer.Forward(sample.Values, sample.Length);
            if (pred == sample.Label) correctAfter++;
        }

        double accAfter = (double)correctAfter / dataset.Count;
        Assert.True(accAfter >= 0.85, $"Transformer accuracy after training should be >= 85%, got: {accAfter * 100:F1}%");
    }

    [Fact]
    public void SimpleLstm_TrainsAndClassifiesSyntheticPatterns()
    {
        int numClasses = 2;
        int seqLen = 32;
        var lstm = new SimpleLstm(numClasses, hiddenDim: 8, seed: 123);

        var dataset = new List<TimeSeriesSample>();
        for (int i = 0; i < 40; i++)
        {
            double[] v0 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) v0[t] = (double)t / seqLen;
            dataset.Add(new TimeSeriesSample(0, seqLen, v0));

            double[] v1 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) v1[t] = 1.0 - ((double)t / seqLen);
            dataset.Add(new TimeSeriesSample(1, seqLen, v1));
        }

        lstm.Train(dataset, epochs: 10, lr: 0.01);

        int correctAfter = 0;
        foreach (var sample in dataset)
        {
            var (pred, _) = lstm.Forward(sample.Values, sample.Length);
            if (pred == sample.Label) correctAfter++;
        }

        double accAfter = (double)correctAfter / dataset.Count;
        Assert.True(accAfter >= 0.85, $"LSTM accuracy after training should be >= 85%, got: {accAfter * 100:F1}%");
    }

    [Fact]
    public void SimpleLstm_ConvergesAtNominalSpeed_AndCollapsesUnderTimeWarp()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "../../../../../datasets/synthetic");
        if (!Directory.Exists(dir)) return;
        var train = DatasetLoader.LoadCsv(Path.Combine(dir, "warpstress_train.csv")).Take(400).ToList();
        var test10 = DatasetLoader.LoadCsv(Path.Combine(dir, "warpstress_test_1.0x.csv")).Take(200).ToList();
        var test05 = DatasetLoader.LoadCsv(Path.Combine(dir, "warpstress_test_0.5x.csv")).Take(200).ToList();
        var test20 = DatasetLoader.LoadCsv(Path.Combine(dir, "warpstress_test_2.0x.csv")).Take(200).ToList();

        var lstm = new SimpleLstm(4, hiddenDim: 16);
        lstm.Train(train, epochs: 10, lr: 0.01);

        double acc10 = (double)test10.Count(s => lstm.Forward(s.Values, s.Length).Prediction == s.Label) / test10.Count;
        double acc05 = (double)test05.Count(s => lstm.Forward(s.Values, s.Length).Prediction == s.Label) / test05.Count;
        double acc20 = (double)test20.Count(s => lstm.Forward(s.Values, s.Length).Prediction == s.Label) / test20.Count;

        // Recurrent baseline converges successfully at nominal speed (>= 90%)
        Assert.True(acc10 >= 0.90, $"LSTM nominal accuracy was {acc10:P1} (expected >= 90%)");

        // Under temporal dilation, Chronos baseline collapses toward chance level (< 40%)
        Assert.True(acc05 < 0.40 || acc20 < 0.40, $"LSTM should collapse under warp, but got 0.5x={acc05:P1}, 2.0x={acc20:P1}");
    }
}
