using System.Text;

namespace Kairos103.Core.Evaluation;

/// <summary>
/// Detailed evaluation metrics report for multiclass classification, focused on Macro F1.
/// </summary>
public sealed record MacroF1Result(
    int ClassCount,
    int TotalSamples,
    double Accuracy,
    double MacroPrecision,
    double MacroRecall,
    double MacroF1,
    double[] ClassPrecision,
    double[] ClassRecall,
    double[] ClassF1,
    int[] ClassSupport,
    int[,] ConfusionMatrix)
{
    private static readonly string[] ClassNames = ["Normal (0)", "SVEB (1)", "VEB (2)", "Fusion (3)"];

    /// <summary>
    /// Formats the metrics as an ASCII classification report table including confusion matrix.
    /// </summary>
    public string FormatReport()
    {
        var sb = new StringBuilder();
        sb.AppendLine("==========================================================================");
        sb.AppendLine("                     CLASSIFICATION METRICS REPORT                        ");
        sb.AppendLine("==========================================================================");
        sb.AppendLine(string.Format("{0,-16} {1,12} {2,12} {3,12} {4,10}", "Class", "Precision", "Recall", "F1-Score", "Support"));
        sb.AppendLine(new string('-', 74));

        for (int c = 0; c < ClassCount; c++)
        {
            string name = c < ClassNames.Length ? ClassNames[c] : $"Class {c}";
            sb.AppendLine(string.Format("{0,-16} {1,12:F4} {2,12:F4} {3,12:F4} {4,10}",
                name, ClassPrecision[c], ClassRecall[c], ClassF1[c], ClassSupport[c]));
        }

        sb.AppendLine(new string('-', 74));
        sb.AppendLine(string.Format("{0,-16} {1,12:F4} {2,12:F4} {3,12:F4} {4,10}",
            "Macro Average", MacroPrecision, MacroRecall, MacroF1, TotalSamples));
        sb.AppendLine(string.Format("{0,-16} {1,38:F4} {2,10}",
            "Accuracy", Accuracy, TotalSamples));
        sb.AppendLine("==========================================================================");
        sb.AppendLine();
        sb.AppendLine("CONFUSION MATRIX (Rows: Actual, Columns: Predicted):");
        sb.Append(string.Format("{0,-14}", "Actual \\ Pred"));
        for (int c = 0; c < ClassCount; c++)
        {
            sb.Append(string.Format("{0,10}", $"Pred {c}"));
        }
        sb.AppendLine();
        sb.AppendLine(new string('-', 14 + ClassCount * 10));

        for (int r = 0; r < ClassCount; r++)
        {
            string rowName = r < ClassNames.Length ? ClassNames[r] : $"Class {r}";
            sb.Append(string.Format("{0,-14}", rowName));
            for (int c = 0; c < ClassCount; c++)
            {
                sb.Append(string.Format("{0,10}", ConfusionMatrix[r, c]));
            }
            sb.AppendLine();
        }
        sb.AppendLine("==========================================================================");
        return sb.ToString();
    }
}

/// <summary>
/// Metric calculator for exact Macro F1 and multiclass confusion matrix.
/// </summary>
public static class MacroF1Metrics
{
    /// <summary>
    /// Computes Macro F1 and per-class metrics from actual and predicted label sequences.
    /// </summary>
    public static MacroF1Result Compute(IReadOnlyList<int> actual, IReadOnlyList<int> predicted, int classCount = 4)
    {
        if (actual.Count != predicted.Count)
            throw new ArgumentException("Actual and predicted sequences must have identical lengths.");
        if (actual.Count == 0)
            throw new ArgumentException("Sequences cannot be empty.");

        int[,] cm = new int[classCount, classCount];
        for (int i = 0; i < actual.Count; i++)
        {
            int a = actual[i];
            int p = predicted[i];
            if (a >= 0 && a < classCount && p >= 0 && p < classCount)
            {
                cm[a, p]++;
            }
        }

        return Compute(cm);
    }

    /// <summary>
    /// Computes Macro F1 and per-class metrics from an existing confusion matrix.
    /// </summary>
    public static MacroF1Result Compute(int[,] cm)
    {
        int classCount = cm.GetLength(0);
        if (cm.GetLength(1) != classCount)
            throw new ArgumentException("Confusion matrix must be square.");

        int totalSamples = 0;
        int correctSamples = 0;
        int[] support = new int[classCount];
        int[] predTotals = new int[classCount];

        for (int r = 0; r < classCount; r++)
        {
            for (int c = 0; c < classCount; c++)
            {
                int val = cm[r, c];
                totalSamples += val;
                support[r] += val;
                predTotals[c] += val;
                if (r == c) correctSamples += val;
            }
        }

        double[] precision = new double[classCount];
        double[] recall = new double[classCount];
        double[] f1 = new double[classCount];

        double sumP = 0.0, sumR = 0.0, sumF1 = 0.0;

        for (int c = 0; c < classCount; c++)
        {
            int tp = cm[c, c];
            int fp = predTotals[c] - tp;
            int fn = support[c] - tp;

            precision[c] = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
            recall[c] = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;

            double prSum = precision[c] + recall[c];
            f1[c] = (prSum > 1e-12) ? (2.0 * precision[c] * recall[c]) / prSum : 0.0;

            sumP += precision[c];
            sumR += recall[c];
            sumF1 += f1[c];
        }

        double accuracy = totalSamples > 0 ? (double)correctSamples / totalSamples : 0.0;
        double macroPrecision = sumP / classCount;
        double macroRecall = sumR / classCount;
        double macroF1 = sumF1 / classCount;

        return new MacroF1Result(
            ClassCount: classCount,
            TotalSamples: totalSamples,
            Accuracy: accuracy,
            MacroPrecision: macroPrecision,
            MacroRecall: macroRecall,
            MacroF1: macroF1,
            ClassPrecision: precision,
            ClassRecall: recall,
            ClassF1: f1,
            ClassSupport: support,
            ConfusionMatrix: cm);
    }
}
