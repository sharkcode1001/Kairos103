namespace Kairos103.Core.Encoding;

/// <summary>
/// Mapeo combinatorio inyectivo B103.
/// Implementa la conversión biyectiva entre permutaciones locales de tamaño N (por defecto N=4 -> 4! = 24)
/// y un índice entero compacto (código de Lehmer en base factorial).
/// </summary>
public static class CombinatorialTable
{
    private static readonly int[] Factorials = [1, 1, 2, 6, 24, 120, 720];

    /// <summary>
    /// Convierte una permutación de elementos en su índice biyectivo de Lehmer (0 a N! - 1).
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
    /// Reconstruye la permutación original a partir de su índice de Lehmer.
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
    /// Retorna el total de estados combinatorios para subregiones de tamaño N (N! = 24 para N=4).
    /// </summary>
    public static int GetPermutationCount(int n = 4) => Factorials[n];
}
