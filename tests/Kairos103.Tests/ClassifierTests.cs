using Kairos103.Core.Data;
using Kairos103.Core.Models;
using Xunit;

namespace Kairos103.Tests;

public class ClassifierTests
{
    [Fact]
    public void KairosClassifier_FitAndPredict_ConvergesOnSeparablePatterns()
    {
        int seqLen = 40;
        var signals = new List<double[]>();
        var labels = new List<int>();

        for (int i = 0; i < 30; i++)
        {
            // Class 0: Sine wave
            double[] s0 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) s0[t] = Math.Sin(2 * Math.PI * t / 20.0);
            signals.Add(s0);
            labels.Add(0);

            // Class 1: Square wave
            double[] s1 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) s1[t] = (t % 10 < 5) ? 1.0 : -1.0;
            signals.Add(s1);
            labels.Add(1);
        }

        var classifier = new KairosClassifier(subregionCount: 8, seed: 42);
        classifier.Fit(signals, labels, epochs: 20, lr: 0.05);

        Assert.True(classifier.IsTrained);
        Assert.Equal(2, classifier.ClassCount);

        // Evaluate predictions
        int[] preds = classifier.PredictBatch(signals);
        int correct = 0;
        for (int i = 0; i < signals.Count; i++)
        {
            if (preds[i] == labels[i]) correct++;
        }

        double accuracy = (double)correct / signals.Count;
        Assert.True(accuracy >= 0.90, $"Expected accuracy >= 0.90, got {accuracy}");
    }

    [Fact]
    public void KairosClassifier_SaveAndLoad_PreservesWeightsAndPredictions()
    {
        int seqLen = 32;
        var samples = new List<SignalSample>();

        for (int i = 0; i < 20; i++)
        {
            double[] s0 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) s0[t] = (double)t / seqLen;
            samples.Add(new SignalSample(0, s0, seqLen));

            double[] s1 = new double[seqLen];
            for (int t = 0; t < seqLen; t++) s1[t] = 1.0 - ((double)t / seqLen);
            samples.Add(new SignalSample(1, s1, seqLen));
        }

        var model = new KairosClassifier(subregionCount: 8, seed: 12345);
        model.Fit(samples, epochs: 15, lr: 0.05);

        string tempPath = Path.Combine(Path.GetTempPath(), $"kairos_test_{Guid.NewGuid():N}.kairos");
        try
        {
            model.Save(tempPath);
            Assert.True(File.Exists(tempPath));

            var loaded = KairosClassifier.Load(tempPath);
            Assert.True(loaded.IsTrained);
            Assert.Equal(model.SubregionCount, loaded.SubregionCount);
            Assert.Equal(model.ClassCount, loaded.ClassCount);
            Assert.Equal(model.TotalFeatures, loaded.TotalFeatures);

            // Compare predictions
            foreach (var s in samples)
            {
                int p1 = model.Predict(s.Values, s.EffectiveLength);
                int p2 = loaded.Predict(s.Values, s.EffectiveLength);
                Assert.Equal(p1, p2);
            }
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void SignalDataset_LoadCsv_HandlesBothFormats()
    {
        string tempKairos = Path.Combine(Path.GetTempPath(), $"kairos_fmt_{Guid.NewGuid():N}.csv");
        string tempStd = Path.Combine(Path.GetTempPath(), $"std_fmt_{Guid.NewGuid():N}.csv");

        try
        {
            // Format 1: Kairos (label,length,v0,v1,v2)
            File.WriteAllText(tempKairos, "label,length,v0,v1,v2\n0,3,0.1,0.2,0.3\n1,3,0.9,0.8,0.7\n");
            var samples1 = SignalDataset.LoadCsv(tempKairos);
            Assert.Equal(2, samples1.Count);
            Assert.Equal(0, samples1[0].Label);
            Assert.Equal(3, samples1[0].EffectiveLength);
            Assert.Equal(3, samples1[0].Values.Length);

            // Format 2: Standard (label,v0,v1,v2) without length column
            File.WriteAllText(tempStd, "label,v0,v1,v2\n0,0.1,0.2,0.3\n1,0.9,0.8,0.7\n");
            var samples2 = SignalDataset.LoadCsv(tempStd);
            Assert.Equal(2, samples2.Count);
            Assert.Equal(0, samples2[0].Label);
            Assert.Equal(3, samples2[0].EffectiveLength);
            Assert.Equal(3, samples2[0].Values.Length);
        }
        finally
        {
            if (File.Exists(tempKairos)) File.Delete(tempKairos);
            if (File.Exists(tempStd)) File.Delete(tempStd);
        }
    }
}
