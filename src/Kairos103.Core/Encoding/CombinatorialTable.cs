namespace Kairos103.Core.Encoding;

/// <summary>
/// Injective combinatorial mapping B103.
/// Implements the bijective mapping between local permutations of size N (default N=4 -> 4! = 24)
/// and a compact integer index (Lehmer code in factorial number system).
/// </summary>
public static class CombinatorialTable
{
    private static readonly int[] Factorials = [1, 1, 2, 6, 24, 120, 720];

    /// <summary>
    /// Converts a permutation of elements into its bijective Lehmer index (0 to N! - 1).
    /// </summary>
    public static int PermutationToLehmerIndex(ReadOnlySpan<int> permutation)
    {
        int n = permutation.Length;
        Span<int> lehmer = stackalloc int[n];

        for (int i = 0; i < n; i++)
        {
            int count = 0;
            for (int j = i + 1; j < n; j++)
            {
                if (permutation[j] < permutation[i])
                    count++;
            }
            lehmer[i] = count;
        }

        int index = 0;
        for (int i = 0; i < n; i++)
        {
            index += lehmer[i] * Factorials[n - 1 - i];
        }

        return index;
    }

    /// <summary>
    /// Reconstructs the original permutation from its Lehmer index.
    /// </summary>
    public static int[] LehmerIndexToPermutation(int index, int n = 4)
    {
        if (n <= 0 || n >= Factorials.Length)
            throw new ArgumentOutOfRangeException(nameof(n));

        var available = Enumerable.Range(0, n).ToList();
        int[] permutation = new int[n];

        for (int i = 0; i < n; i++)
        {
            int fact = Factorials[n - 1 - i];
            int lehmerDigit = index / fact;
            index %= fact;

            permutation[i] = available[lehmerDigit];
            available.RemoveAt(lehmerDigit);
        }

        return permutation;
    }

    /// <summary>
    /// Returns the total combinatorial states for subregions of size N (N! = 24 for N=4).
    /// </summary>
    public static int GetPermutationCount(int n = 4) => Factorials[n];
}
