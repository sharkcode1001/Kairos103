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

        // Generar un dataset sintético separable
        var dataset = new List<TimeSeriesSample>();
        for (int i = 0; i < 40; i++)
        {
            // Clase 0: rampa ascendente
            double[] v0 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) v0[t] = (double)t / seqLen;
            dataset.Add(new TimeSeriesSample(0, seqLen, v0));

            // Clase 1: rampa descendente
            double[] v1 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) v1[t] = 1.0 - ((double)t / seqLen);
            dataset.Add(new TimeSeriesSample(1, seqLen, v1));
        }

        // Evaluar antes de entrenar
        int correctBefore = 0;
        foreach (var sample in dataset)
        {
            var (pred, _) = transformer.Forward(sample.Values, sample.Length);
            if (pred == sample.Label) correctBefore++;
        }

        // Entrenar con el algoritmo Adam implementado
        transformer.Train(dataset, epochs: 10, lr: 0.01);

        // Evaluar después de entrenar
        int correctAfter = 0;
        foreach (var sample in dataset)
        {
            var (pred, _) = transformer.Forward(sample.Values, sample.Length);
            if (pred == sample.Label) correctAfter++;
        }

        double accAfter = (double)correctAfter / dataset.Count;
        Assert.True(accAfter >= 0.85, $"La precisión del Transformer después de entrenar debe ser >= 85%, obtenida: {accAfter * 100:F1}%");
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
        Assert.True(accAfter >= 0.85, $"La precisión de la LSTM después de entrenar debe ser >= 85%, obtenida: {accAfter * 100:F1}%");
    }
}
