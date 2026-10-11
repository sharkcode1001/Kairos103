namespace Kairos103.Core.Models;

/// <summary>
/// Clasificador lineal supervisado con aprendizaje discriminativo para vectores dispersos de rango.
/// </summary>
public sealed class PairwiseTrainer
{
    public int ClassCount { get; }
    public int FeatureDim { get; }

    // Matriz de pesos: ClassCount x FeatureDim
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
    /// Entrena el clasificador con descenso por gradiente estocástico sobre vectores dispersos.
    /// </summary>
    public void Train(IReadOnlyList<(double[] Features, int Label)> dataset, int epochs = 20, double lr = 0.05)
    {
        double[] scores = new double[ClassCount];

        for (int ep = 0; ep < epochs; ep++)
        {
            foreach (var (features, label) in dataset)
            {
                // Inferencia y cálculo de puntajes
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

                // Predicción argmax
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

                // Actualización perceptrón / margen
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
    /// Infiere la clase predicha para un vector de características y cuenta las operaciones requeridas.
    /// Para características one-hot/dispersas, solo se realizan sumas de acumulación.
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
                    score += Weights[c][f];
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
