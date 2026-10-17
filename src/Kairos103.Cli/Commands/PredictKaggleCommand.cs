using System.Diagnostics;
using Kairos103.Core.Data;
using Kairos103.Core.Models;

namespace Kairos103.Cli.Commands;

public static class PredictKaggleCommand
{
    public static void Execute(
        string modelPath,
        string testPath,
        string outputPath,
        string? sampleSubmissionPath = null,
        double[]? customThresholds = null)
    {
        Console.WriteLine("\n==========================================================================");
        Console.WriteLine("          KAIROS103 - KAGGLE SUBMISSION GENERATOR                         ");
        Console.WriteLine("==========================================================================");

        string[] modelFiles = modelPath.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (modelFiles.Length == 0)
        {
            Console.WriteLine($"[ERROR] No valid model paths specified: {modelPath}");
            return;
        }

        var classifiers = new List<KairosClassifier>();
        foreach (var file in modelFiles)
        {
            if (!File.Exists(file))
            {
                Console.WriteLine($"[ERROR] Model file not found: {file}");
                return;
            }
            Console.WriteLine($"Loading model from: {file}...");
            var c = KairosClassifier.Load(file);
            string calibInfo = (c.CalibScale != null && c.CalibScale.Length == c.ClassCount) ? " [Platt Calibrated]" : "";
            Console.WriteLine($"  Loaded Kairos103 model (Subregions: {c.SubregionCount}, Classes: {c.ClassCount}, Readout Dim: {c.HiddenDim}){calibInfo}");
            classifiers.Add(c);
        }

        // Validate ensemble models share matching classification head configuration
        for (int m = 1; m < classifiers.Count; m++)
        {
            if (classifiers[m].ClassCount != classifiers[0].ClassCount ||
                classifiers[m].IsOvr != classifiers[0].IsOvr)
            {
                throw new InvalidOperationException(
                    $"Ensemble architecture mismatch between '{modelFiles[0]}' and '{modelFiles[m]}'! " +
                    $"Classes ({classifiers[0].ClassCount} vs {classifiers[m].ClassCount}), " +
                    $"IsOvr ({classifiers[0].IsOvr} vs {classifiers[m].IsOvr}) must be identical.");
            }
        }

        if (!File.Exists(testPath))
        {
            Console.WriteLine($"[ERROR] Test data file not found: {testPath}");
            return;
        }

        double[] activeThresholds;
        if (customThresholds != null && customThresholds.Length == classifiers[0].ClassCount)
        {
            activeThresholds = customThresholds;
            Console.WriteLine($"  Overriding decision thresholds with user-specified: [{string.Join(", ", activeThresholds.Select((t, c) => $"Class {c}={t:F2}"))}]");
        }
        else
        {
            activeThresholds = new double[classifiers[0].ClassCount];
            int countWithThresh = 0;
            for (int m = 0; m < classifiers.Count; m++)
            {
                if (classifiers[m].Thresholds != null && classifiers[m].Thresholds.Length == classifiers[0].ClassCount)
                {
                    for (int c = 0; c < classifiers[0].ClassCount; c++)
                    {
                        activeThresholds[c] += classifiers[m].Thresholds[c];
                    }
                    countWithThresh++;
                }
            }

            if (countWithThresh > 0)
            {
                for (int c = 0; c < classifiers[0].ClassCount; c++)
                {
                    activeThresholds[c] /= countWithThresh;
                }
                string srcDesc = countWithThresh == 1 ? "model-calibrated" : $"ensemble average calibrated ({countWithThresh} models)";
                Console.WriteLine($"  Applying {srcDesc} decision thresholds: [{string.Join(", ", activeThresholds.Select((t, c) => $"Class {c}={t:F2}"))}]");
            }
            else
            {
                activeThresholds = [1.00, 1.10, 1.00, 1.00];
                Console.WriteLine($"  Applying default decision thresholds: [{string.Join(", ", activeThresholds.Select((t, c) => $"Class {c}={t:F2}"))}]");
            }
        }

        double[]? activeBias = null;
        if (customThresholds == null)
        {
            int classCount = classifiers[0].ClassCount;
            var biasSum = new double[classCount];
            int countWithBias = 0;
            for (int m = 0; m < classifiers.Count; m++)
            {
                var b = classifiers[m].DecisionBias;
                if (b != null && b.Length == classCount)
                {
                    for (int c = 0; c < classCount; c++) biasSum[c] += b[c];
                    countWithBias++;
                }
            }
            if (countWithBias > 0)
            {
                for (int c = 0; c < classCount; c++) biasSum[c] /= countWithBias;
                activeBias = biasSum;
                Console.WriteLine($"  Applying additive logit biases (ensemble avg of {countWithBias} models): [{string.Join(", ", activeBias.Select((b, c) => $"Class {c}={b:F3}"))}]");
            }
        }
        else
        {
            Console.WriteLine("  [INFO] User-specified custom thresholds take precedence over model decision bias.");
        }

        if (classifiers.Count == 1)
        {
            classifiers[0].Thresholds = activeThresholds;
        }

        Console.WriteLine($"Loading test dataset from: {testPath}...");
        var sw = Stopwatch.StartNew();
        var testSamples = KaggleEcgDataset.LoadTest(testPath);
        sw.Stop();
        Console.WriteLine($"Loaded {testSamples.Count:N0} test samples in {sw.ElapsedMilliseconds} ms.");

        // If sample_submission is provided, check ID alignment
        List<string>? expectedIds = null;
        if (!string.IsNullOrEmpty(sampleSubmissionPath) && File.Exists(sampleSubmissionPath))
        {
            Console.WriteLine($"Verifying against sample submission: {sampleSubmissionPath}...");
            expectedIds = KaggleEcgDataset.LoadSampleSubmissionIds(sampleSubmissionPath);
            if (expectedIds.Count != testSamples.Count)
            {
                throw new InvalidOperationException($"Row count mismatch! sample_submission has {expectedIds.Count} rows, but test.csv has {testSamples.Count} rows.");
            }
            Console.WriteLine($"  Sample submission alignment verified: {expectedIds.Count:N0} IDs match.");
        }

        // Run batch inference
        if (classifiers.Count > 1)
        {
            Console.WriteLine($"\nRunning neuromorphic ensemble inference across {classifiers.Count} models...");
        }
        else
        {
            Console.WriteLine("\nRunning neuromorphic inference...");
        }

        sw.Restart();
        var predictions = new int[testSamples.Count];
        int[] classTally = new int[4];
        double[] ensembleProbs = new double[4];
        int predReportStep = Math.Max(1000, testSamples.Count / 10);

        for (int i = 0; i < testSamples.Count; i++)
        {
            if (testSamples.Count >= 2000 && (i % predReportStep == 0 || i == testSamples.Count - 1))
            {
                double pct = (double)(i + 1) / testSamples.Count * 100.0;
                Console.WriteLine($"  [Inference] Processed {i + 1:N0} / {testSamples.Count:N0} test samples ({pct:F1}%)...");
                Console.Out.Flush();
            }

            var s = testSamples[i];
            var combinedSignal = s.ToCombinedSignal();

            int pred;
            if (classifiers.Count == 1)
            {
                pred = classifiers[0].Predict(combinedSignal, effectiveLength: KaggleEcgSample.WaveformLength);
            }
            else if (activeBias != null)
            {
                Array.Clear(ensembleProbs, 0, 4);
                for (int m = 0; m < classifiers.Count; m++)
                {
                    double[] logits = classifiers[m].PredictCalibratedLogits(combinedSignal, effectiveLength: KaggleEcgSample.WaveformLength);
                    for (int c = 0; c < 4; c++)
                    {
                        ensembleProbs[c] += logits[c];
                    }
                }

                double invM = 1.0 / classifiers.Count;
                for (int c = 0; c < 4; c++)
                {
                    ensembleProbs[c] = ensembleProbs[c] * invM + activeBias[c];
                }

                int bestBias = 0;
                double bestBiasScore = ensembleProbs[0];
                for (int c = 1; c < 4; c++)
                {
                    if (ensembleProbs[c] > bestBiasScore)
                    {
                        bestBiasScore = ensembleProbs[c];
                        bestBias = c;
                    }
                }
                pred = bestBias;
            }
            else
            {
                Array.Clear(ensembleProbs, 0, 4);
                for (int m = 0; m < classifiers.Count; m++)
                {
                    double[] probs = classifiers[m].PredictProbabilities(combinedSignal, effectiveLength: KaggleEcgSample.WaveformLength);
                    for (int c = 0; c < 4; c++)
                    {
                        ensembleProbs[c] += probs[c];
                    }
                }

                // Normalize accumulated probabilities to [0, 1] range
                double invM = 1.0 / classifiers.Count;
                for (int c = 0; c < 4; c++)
                {
                    ensembleProbs[c] *= invM;
                }

                int best = 0;
                double bestScore = ensembleProbs[0] / Math.Max(1e-4, activeThresholds[0]);
                for (int c = 1; c < 4; c++)
                {
                    double score = ensembleProbs[c] / Math.Max(1e-4, activeThresholds[c]);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        best = c;
                    }
                }
                pred = best;
            }

            if (pred < 0 || pred > 3)
            {
                throw new InvalidOperationException($"Predicted label {pred} at sample index {i} is out of bounds (must be 0, 1, 2, or 3).");
            }
            predictions[i] = pred;
            classTally[pred]++;
        }
        sw.Stop();
        Console.Out.Flush();

        Console.WriteLine($"Inference finished in {sw.ElapsedMilliseconds} ms ({(double)sw.ElapsedMilliseconds / testSamples.Count * 1000.0:F2} µs/sample).");

        // Report predicted distribution
        Console.WriteLine("\nPredicted Class Distribution for Kaggle Submission:");
        string[] classNames = ["0 (Normal)", "1 (SVEB)", "2 (VEB)", "3 (Fusion)"];
        for (int c = 0; c < 4; c++)
        {
            double pct = (double)classTally[c] / testSamples.Count * 100.0;
            Console.WriteLine($"  Class {classNames[c],-12}: {classTally[c],7:N0} ({pct,5:F2}%)");
        }

        // Ensure target directory exists
        string? outDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outDir) && !Directory.Exists(outDir))
        {
            Directory.CreateDirectory(outDir);
        }

        // Write submission CSV
        Console.WriteLine($"\nWriting Kaggle submission to: {outputPath}...");
        using (var writer = new StreamWriter(outputPath, append: false))
        {
            writer.WriteLine("id,label");
            for (int i = 0; i < testSamples.Count; i++)
            {
                string id = (expectedIds != null && i < expectedIds.Count) ? expectedIds[i] : testSamples[i].Id;
                writer.WriteLine($"{id},{predictions[i]}");
            }
        }

        // Self-validation check of generated submission file
        ValidateGeneratedSubmission(outputPath, testSamples.Count);

        Console.WriteLine($"\n[SUCCESS] Submission ready for Kaggle! File: {outputPath}");
        Console.WriteLine("==========================================================================\n");
    }

    private static void ValidateGeneratedSubmission(string filePath, int expectedRows)
    {
        Console.WriteLine("Performing automated integrity validation of submission file...");
        using var reader = new StreamReader(filePath);
        string? header = reader.ReadLine();
        if (header != "id,label")
        {
            throw new InvalidOperationException($"Invalid submission header: '{header}'. Expected exactly 'id,label'.");
        }

        int count = 0;
        var seenIds = new HashSet<string>(expectedRows);
        int[] classCounts = new int[4];
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            count++;
            var parts = line.Split(',');
            if (parts.Length != 2)
            {
                throw new InvalidOperationException($"Row {count + 1} has {parts.Length} columns. Expected exactly 2 (id,label). Line: '{line}'");
            }

            string id = parts[0].Trim();
            if (!seenIds.Add(id))
            {
                throw new InvalidOperationException($"Duplicate sample ID detected at row {count + 1}: '{id}'");
            }

            if (!int.TryParse(parts[1].Trim(), out int label) || label < 0 || label > 3)
            {
                throw new InvalidOperationException($"Row {count + 1} has invalid label value '{parts[1]}'. Must be integer 0, 1, 2, or 3.");
            }
            classCounts[label]++;
        }

        if (count != expectedRows)
        {
            throw new InvalidOperationException($"Row count mismatch in submission! Expected {expectedRows} rows, but file has {count} rows.");
        }

        for (int c = 0; c < 4; c++)
        {
            if (classCounts[c] == 0)
            {
                Console.WriteLine($"  [WARNING] Class {c} has 0 predicted samples in submission! Review thresholds to avoid Macro F1 collapse.");
            }
        }

        Console.WriteLine($"  Validation passed! Header is 'id,label', exactly {count:N0} unique IDs, all labels in [0, 3].");
    }
}
