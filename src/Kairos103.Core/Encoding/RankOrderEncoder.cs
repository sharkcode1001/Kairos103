namespace Kairos103.Core.Encoding;

/// <summary>
/// Codificador de orden de llegada (Rank-Order Encoder).
/// Convierte las permutaciones relativas de primer disparo en ponderaciones dispersas w_r = 1 / rango.
/// </summary>
public static class RankOrderEncoder
{
    /// <summary>
    /// Calcula los pesos de orden de llegada para una permutación dada.
    /// El primer disparo (rango 1) recibe peso 1.0, el segundo 0.5 (1/2), el tercero 0.333 (1/3), etc.
    /// Si un canal no disparó (tiempo infinito), su peso asignado es 0.0.
    /// </summary>
    public static double[] EncodeRanks(IReadOnlyList<double> spikeTimes)
    {
        int n = spikeTimes.Count;
        double[] weights = new double[n];

        var orderedIndices = new List<int>(n);
        for (int i = 0; i < n; i++)
        {
            if (!double.IsPositiveInfinity(spikeTimes[i]))
            {
                orderedIndices.Add(i);
            }
        }

        orderedIndices.Sort((a, b) => spikeTimes[a].CompareTo(spikeTimes[b]));

        for (int rank = 0; rank < orderedIndices.Count; rank++)
        {
            int channelIndex = orderedIndices[rank];
            weights[channelIndex] = 1.0 / (rank + 1);
        }

        return weights;
    }

    /// <summary>
    /// Retorna la permutación de índices ordenados por tiempo de disparo.
    /// </summary>
    public static int[] GetPermutation(IReadOnlyList<double> spikeTimes)
    {
        int n = spikeTimes.Count;
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        Array.Sort(indices, (a, b) => spikeTimes[a].CompareTo(spikeTimes[b]));
        return indices;
    }
}
