namespace Kairos103.Core.Encoding;

/// <summary>
/// Rank-Order Encoder.
/// Converts relative first-spike arrival permutations into sparse weights: w_r = 1 / rank.
/// </summary>
public static class RankOrderEncoder
{
    /// <summary>
    /// Computes rank-order weights for a given set of first-spike arrival times.
    /// The first spike (rank 1) receives weight 1.0, the second 0.5 (1/2), the third 0.333 (1/3), etc.
    /// Non-firing channels (infinite timestamp) receive a weight of 0.0.
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
    /// Returns the permutation array of channel indices ordered by spike arrival timestamp.
    /// Uses a stable insertion sort so tied arrival times preserve ascending channel index order.
    /// </summary>
    public static int[] GetPermutation(IReadOnlyList<double> spikeTimes)
    {
        int n = spikeTimes.Count;
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        for (int i = 1; i < n; i++)
        {
            int key = indices[i];
            double keyTime = spikeTimes[key];
            int j = i - 1;
            while (j >= 0 && spikeTimes[indices[j]] > keyTime)
            {
                indices[j + 1] = indices[j];
                j--;
            }
            indices[j + 1] = key;
        }

        return indices;
    }
}
