using Kairos103.Core.Encoding;
using Xunit;

namespace Kairos103.Tests;

public class CombinatorialTests
{
    [Fact]
    public void LehmerMapping_IsBijectiveForSubregionsOfFourNeurons()
    {
        int n = 4;
        int totalPermutations = CombinatorialTable.GetPermutationCount(n); // 4! = 24
        Assert.Equal(24, totalPermutations);

        var seenIndices = new HashSet<int>();

        // Generar todas las 24 permutaciones posibles de 0, 1, 2, 3
        var allPermutations = GenerateAllPermutations([0, 1, 2, 3]);
        Assert.Equal(24, allPermutations.Count);

        foreach (var p in allPermutations)
        {
            int index = CombinatorialTable.PermutationToLehmerIndex(p);

            // Debe estar en el rango [0, 23]
            Assert.InRange(index, 0, 23);

            // Debe ser inyectivo (único)
            Assert.True(seenIndices.Add(index), $"Índice colisionado: {index}");

            // Debe ser invertible (biyectivo)
            int[] reconstructed = CombinatorialTable.LehmerIndexToPermutation(index, n);
            Assert.Equal(p, reconstructed);
        }

        // Se cubrieron todos los 24 enteros posibles
        Assert.Equal(24, seenIndices.Count);
    }

    private static List<int[]> GenerateAllPermutations(int[] items)
    {
        var result = new List<int[]>();
        Permute(items, 0, items.Length - 1, result);
        return result;
    }

    private static void Permute(int[] arr, int start, int end, List<int[]> result)
    {
        if (start == end)
        {
            result.Add((int[])arr.Clone());
            return;
        }

        for (int i = start; i <= end; i++)
        {
            (arr[start], arr[i]) = (arr[i], arr[start]);
            Permute(arr, start + 1, end, result);
            (arr[start], arr[i]) = (arr[i], arr[start]);
        }
    }
}
