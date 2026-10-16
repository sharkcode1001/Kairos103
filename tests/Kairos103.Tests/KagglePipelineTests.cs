using System.Globalization;
using Kairos103.Core.Baselines;
using Kairos103.Core.Data;
using Kairos103.Core.Evaluation;
using Kairos103.Core.Models;
using Xunit;

namespace Kairos103.Tests;

public class KagglePipelineTests
{
    [Fact]
    public void MacroF1_ComputesExpectedValue_OnKnownMatrix()
    {
        // 4x4 known confusion matrix
        // Rows: Actual, Cols: Predicted
        int[,] cm = new int[,]
        {
            { 80,  5,  5, 10 }, // Class 0: TP=80,  FN=20, Support=100. FP = 10+5+0 = 15. Prec = 80/95, Rec = 80/100
            { 10, 40, 10,  0 }, // Class 1: TP=40,  FN=20, Support=60.  FP = 5+5+10 = 20. Prec = 40/60, Rec = 40/60
            {  5,  5, 30,  0 }, // Class 2: TP=30,  FN=10, Support=40.  FP = 5+10+0 = 15. Prec = 30/45, Rec = 30/40
            {  0, 10,  0, 20 }  // Class 3: TP=20,  FN=10, Support=30.  FP = 10+0+0 = 10. Prec = 20/30, Rec = 20/30
        };

        var res = MacroF1Metrics.Compute(cm);

        Assert.Equal(4, res.ClassCount);
        Assert.Equal(230, res.TotalSamples);

        // Check Class 1: Prec = 40/60 = 0.6667, Rec = 40/60 = 0.6667 -> F1 = 0.6667
        Assert.Equal(40.0 / 60.0, res.ClassPrecision[1], 4);
        Assert.Equal(40.0 / 60.0, res.ClassRecall[1], 4);
        Assert.Equal(40.0 / 60.0, res.ClassF1[1], 4);

        // Macro F1 must be average of the 4 F1 scores
        double expectedMacroF1 = (res.ClassF1[0] + res.ClassF1[1] + res.ClassF1[2] + res.ClassF1[3]) / 4.0;
        Assert.Equal(expectedMacroF1, res.MacroF1, 4);

        // Check report string formatting
        string report = res.FormatReport();
        Assert.Contains("Macro Average", report);
        Assert.Contains("CONFUSION MATRIX", report);
    }

    [Fact]
    public void KaggleDataset_ParsesCorrectly_AndValidatesBounds()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            string trainCsv = Path.Combine(tempDir, "train.csv");
            string testCsv = Path.Combine(tempDir, "test.csv");
            string sampleCsv = Path.Combine(tempDir, "sample_submission.csv");

            // Build small train.csv: id, 250 signals, pre_rr, post_rr, rr_ratio, label
            using (var writer = new StreamWriter(trainCsv))
            {
                var sigHeaders = string.Join(",", Enumerable.Range(0, 250).Select(i => $"sig_{i}"));
                writer.WriteLine($"id,{sigHeaders},pre_rr,post_rr,rr_ratio,label");

                // Row 1: label 0
                var sig1 = string.Join(",", Enumerable.Repeat("0.5", 250));
                writer.WriteLine($"sub_100,{sig1},0.82,0.85,0.96,0");

                // Row 2: label 1
                var sig2 = string.Join(",", Enumerable.Repeat("1.2", 250));
                writer.WriteLine($"sub_101,{sig2},0.45,0.70,0.64,1");

                // Row 3: label 2
                var sig3 = string.Join(",", Enumerable.Repeat("-0.3", 250));
                writer.WriteLine($"sub_102,{sig3},1.10,1.15,0.95,2");

                // Row 4: label 3
                var sig4 = string.Join(",", Enumerable.Repeat("0.0", 250));
                writer.WriteLine($"sub_103,{sig4},0.90,0.92,0.98,3");
            }

            // Build test.csv
            using (var writer = new StreamWriter(testCsv))
            {
                var sigHeaders = string.Join(",", Enumerable.Range(0, 250).Select(i => $"sig_{i}"));
                writer.WriteLine($"id,{sigHeaders},pre_rr,post_rr,rr_ratio");
                var sigT = string.Join(",", Enumerable.Repeat("0.2", 250));
                writer.WriteLine($"test_201,{sigT},0.80,0.85,0.94");
                writer.WriteLine($"test_202,{sigT},0.50,0.65,0.77");
            }

            // Build sample_submission.csv
            using (var writer = new StreamWriter(sampleCsv))
            {
                writer.WriteLine("id,label");
                writer.WriteLine("test_201,0");
                writer.WriteLine("test_202,0");
            }

            // 1. Test LoadTrain
            var trainSamples = KaggleEcgDataset.LoadTrain(trainCsv);
            Assert.Equal(4, trainSamples.Count);
            Assert.Equal("sub_100", trainSamples[0].Id);
            Assert.Equal(0, trainSamples[0].Label);
            Assert.Equal(250, trainSamples[0].Waveform.Length);
            Assert.Equal(0.82, trainSamples[0].PreRr, 3);
            Assert.Equal(0.85, trainSamples[0].PostRr, 3);
            Assert.Equal(0.96, trainSamples[0].RrRatio, 3);

            // SignalSample must have EffectiveLength = 250 and Total = 268 (250 waveform + 18 aux features)
            var sigSample = trainSamples[0].ToSignalSample();
            Assert.Equal(250, sigSample.EffectiveLength);
            Assert.Equal(KaggleEcgSample.TotalLength, sigSample.Values.Length);
            Assert.Equal(0.96, sigSample.Values[252], 3);

            // 2. Test LoadTest
            var testSamples = KaggleEcgDataset.LoadTest(testCsv);
            Assert.Equal(2, testSamples.Count);
            Assert.Equal("test_201", testSamples[0].Id);
            Assert.Equal("test_202", testSamples[1].Id);
            Assert.Equal(-1, testSamples[0].Label);

            // 3. Test LoadSampleSubmissionIds
            var sampleIds = KaggleEcgDataset.LoadSampleSubmissionIds(sampleCsv);
            Assert.Equal(2, sampleIds.Count);
            Assert.Equal("test_201", sampleIds[0]);
            Assert.Equal("test_202", sampleIds[1]);

            // 4. Test Class weights
            var weights = KaggleEcgDataset.ComputeBalancedClassWeights(trainSamples, 4);
            Assert.Equal(4, weights.Length);
            Assert.All(weights, w => Assert.True(w > 0.0));
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void EndToEnd_EcgPipeline_FitsAndPredictsFourClasses()
    {
        // Synthesize 120 artificial ECG samples across 4 classes
        var rng = new Random(1234);
        var samples = new List<KaggleEcgSample>();

        for (int i = 0; i < 120; i++)
        {
            int label = i % 4;
            double[] wave = new double[250];

            // Give each class a distinct wave pattern and RR ratio characteristic
            for (int t = 0; t < 250; t++)
            {
                double phase = (double)t / 250.0 * Math.PI * 2.0;
                wave[t] = label switch
                {
                    0 => Math.Sin(phase * 2.0) + (rng.NextDouble() * 0.1),
                    1 => Math.Sin(phase * 4.0) + (rng.NextDouble() * 0.1),
                    2 => (t is >= 100 and <= 150 ? 2.5 : 0.0) + (rng.NextDouble() * 0.1),
                    3 => (t is >= 80 and <= 170 ? -1.8 : 0.5) + (rng.NextDouble() * 0.1),
                    _ => 0.0
                };
            }

            double preRr = label switch { 0 => 0.9, 1 => 0.5, 2 => 1.1, 3 => 0.8, _ => 1.0 };
            double postRr = label switch { 0 => 0.9, 1 => 0.8, 2 => 0.7, 3 => 0.85, _ => 1.0 };
            double rrRatio = preRr / Math.Max(0.01, postRr);

            samples.Add(new KaggleEcgSample($"id_{i}", wave, preRr, postRr, rrRatio, label));
        }

        // Stratified split: 80% train (96 samples), 20% val (24 samples)
        var (train, val) = KaggleEcgDataset.StratifiedSplit(samples, valFraction: 0.2, seed: 42);
        Assert.Equal(96, train.Count);
        Assert.Equal(24, val.Count);

        // Train KairosClassifier
        var classifier = new KairosClassifier(subregionCount: 8, seed: 42);
        var trainSignals = train.Select(s => s.ToSignalSample()).ToList();
        var valSignals = val.Select(s => s.ToSignalSample()).ToList();

        var weights = KaggleEcgDataset.ComputeBalancedClassWeights(train, 4);
        var bestMetric = classifier.FitWithValidation(trainSignals, valSignals, epochs: 15, lr: 0.05, hiddenDim: 16, classWeights: weights);

        Assert.True(bestMetric.MacroF1 > 0.50, $"Macro F1 was {bestMetric.MacroF1}, expected > 0.50");

        // Baselines evaluation
        var majority = new MajorityDummyClassifier();
        majority.Fit(train);
        int majPred = majority.Predict(val[0]);
        Assert.True(majPred >= 0 && majPred < 4);

        var linear = new LinearEcgClassifier();
        linear.Fit(train);
        int linPred = linear.Predict(val[0]);
        Assert.True(linPred >= 0 && linPred < 4);

        var cnn = new Cnn1DEcgClassifier();
        cnn.Fit(train);
        int cnnPred = cnn.Predict(val[0]);
        Assert.True(cnnPred >= 0 && cnnPred < 4);

        // Verify Save and Load preserve auxiliary features (FeatureDim = TotalFeatures + 3)
        string tempModel = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.kairos");
        try
        {
            classifier.Save(tempModel);
            var loaded = KairosClassifier.Load(tempModel);

            // Subregions: 8 * 24 = 192, plus 18 aux features = 210
            Assert.Equal(8 * 24 + KaggleEcgSample.AuxCount, loaded.TotalFeatures);

            for (int i = 0; i < val.Count; i++)
            {
                int pOriginal = classifier.Predict(val[i].ToCombinedSignal(), effectiveLength: 250);
                int pLoaded = loaded.Predict(val[i].ToCombinedSignal(), effectiveLength: 250);
                Assert.Equal(pOriginal, pLoaded);

                double[] probs = classifier.PredictProbabilities(val[i].ToCombinedSignal(), effectiveLength: 250);
                Assert.Equal(4, probs.Length);
                Assert.Equal(1.0, probs.Sum(), 4);
                int maxProbIdx = Array.IndexOf(probs, probs.Max());
                Assert.Equal(pOriginal, maxProbIdx);
            }
        }
        finally
        {
            if (File.Exists(tempModel)) File.Delete(tempModel);
        }
    }

    [Fact]
    public void OvrReadout_Trains_Saves_Loads_AndPredicts_Successfully()
    {
        var rng = new Random(42);
        var samples = new List<KaggleEcgSample>();

        // Generate 40 synthetic samples with 4 classes
        for (int i = 0; i < 40; i++)
        {
            double[] wave = new double[250];
            int label = i % 4;
            for (int t = 0; t < 250; t++)
            {
                wave[t] = Math.Sin(t * 0.1 * (label + 1)) + rng.NextDouble() * 0.05;
            }
            samples.Add(new KaggleEcgSample($"syn_{i}", wave, 300.0, 320.0, 300.0 / 320.0, label));
        }

        var (train, val) = KaggleEcgDataset.StratifiedSplit(samples, 0.25, seed: 42);
        var trainSignals = train.Select(s => s.ToSignalSample()).ToList();
        var valSignals = val.Select(s => s.ToSignalSample()).ToList();

        var classifier = new KairosClassifier(subregionCount: 4, seed: 42);
        var res = classifier.FitOvrWithValidation(trainSignals, valSignals, epochs: 5, lr: 0.02, hiddenDim: 16);

        Assert.True(classifier.IsOvr);
        Assert.True(res.MacroF1 >= 0.0);

        string tempModel = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}_ovr.kairos");
        try
        {
            classifier.Save(tempModel);
            var loaded = KairosClassifier.Load(tempModel);

            Assert.True(loaded.IsOvr);
            Assert.Equal(classifier.TotalFeatures, loaded.TotalFeatures);

            for (int i = 0; i < val.Count; i++)
            {
                int pOriginal = classifier.Predict(val[i].ToCombinedSignal(), effectiveLength: 250);
                int pLoaded = loaded.Predict(val[i].ToCombinedSignal(), effectiveLength: 250);
                Assert.Equal(pOriginal, pLoaded);

                double[] probs = loaded.PredictProbabilities(val[i].ToCombinedSignal(), effectiveLength: 250);
                Assert.Equal(4, probs.Length);
                foreach (var p in probs)
                {
                    Assert.InRange(p, 0.0, 1.0);
                }
            }
        }
        finally
        {
            if (File.Exists(tempModel)) File.Delete(tempModel);
        }
    }

    [Fact]
    public void StratifiedKFolds_PartitionsCorrectly_PreservingClassFrequencies()
    {
        var samples = new List<KaggleEcgSample>();
        for (int i = 0; i < 100; i++)
        {
            int label = i % 4;
            double[] wave = new double[250];
            samples.Add(new KaggleEcgSample($"id_{i}", wave, 0.8, 0.8, 1.0, label));
        }

        int kFolds = 5;
        var folds = KaggleEcgDataset.StratifiedKFolds(samples, kFolds, seed: 42);

        Assert.Equal(kFolds, folds.Count);
        int totalAcrossFolds = 0;
        foreach (var fold in folds)
        {
            Assert.Equal(20, fold.Count);
            totalAcrossFolds += fold.Count;

            // Each fold should have equal class representation (5 samples per class)
            var classCounts = fold.GroupBy(s => s.Label).ToDictionary(g => g.Key, g => g.Count());
            Assert.Equal(4, classCounts.Count);
            foreach (var kvp in classCounts)
            {
                Assert.Equal(5, kvp.Value);
            }
        }

        Assert.Equal(100, totalAcrossFolds);
    }
}
