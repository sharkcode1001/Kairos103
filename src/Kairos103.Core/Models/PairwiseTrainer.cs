namespace Kairos103.Core.Models;

/// <summary>
/// Supervised linear classifier with discriminative margin learning for sparse rank vectors.
/// </summary>
public sealed class PairwiseTrainer
{
    public int ClassCount { get; }
    public int FeatureDim { get; }

    // Weight matrix: ClassCount x FeatureDim
    public double[][] Weights { get; }

    public PairwiseTrainer(int classCount, int featureDim)
    {
        ClassCount = classCount;
        FeatureDim = featureDim;
        Weights = new double[classCount][];
        for (int c = 0; c < classCount; c++)
        {
            Weights[c] = new double[featureDim];
        }
    }

    /// <summary>
    /// Trains the classifier using stochastic gradient updates on sparse rank features.
    /// </summary>
    public void Train(IReadOnlyList<(double[] Features, int Label)> dataset, int epochs = 20, double lr = 0.05, Action<int, int>? onProgress = null)
    {
        double[] scores = new double[ClassCount];

        for (int ep = 0; ep < epochs; ep++)
        {
            onProgress?.Invoke(ep + 1, epochs);
            foreach (var (features, label) in dataset)
            {
                // Inference and score computation
                Array.Clear(scores);
                for (int c = 0; c < ClassCount; c++)
                {
                    double s = 0.0;
                    for (int f = 0; f < FeatureDim; f++)
                    {
                        if (features[f] > 0.0)
                        {
                            s += Weights[c][f] * features[f];
                        }
                    }
                    scores[c] = s;
                }

                // Argmax prediction
                int predicted = 0;
                double maxScore = scores[0];
                for (int c = 1; c < ClassCount; c++)
                {
                    if (scores[c] > maxScore)
                    {
                        maxScore = scores[c];
                        predicted = c;
                    }
                }

                // Perceptron / margin update
                if (predicted != label)
                {
                    for (int f = 0; f < FeatureDim; f++)
                    {
                        if (features[f] > 0.0)
                        {
                            Weights[label][f] += lr * features[f];
                            Weights[predicted][f] -= lr * features[f];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Infers the predicted class for a feature vector and tallies required operations.
    /// For sparse features, only integer additions and score accumulations are required.
    /// </summary>
    public (int Prediction, int AccumulateOps) Predict(double[] features)
    {
        int bestClass = 0;
        double bestScore = double.NegativeInfinity;
        int accOps = 0;

        for (int c = 0; c < ClassCount; c++)
        {
            double score = 0.0;
            for (int f = 0; f < FeatureDim; f++)
            {
                if (features[f] > 0.0)
                {
                    score += Weights[c][f] * features[f];
                    accOps++;
                }
            }

            if (score > bestScore)
            {
                bestScore = score;
                bestClass = c;
            }
        }

        return (bestClass, accOps);
    }
}
