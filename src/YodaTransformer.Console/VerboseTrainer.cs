namespace YodaTransformer;

public static class VerboseTrainer
{
    private const int BarWidth    = 20;
    private const int PairLabelW  = 24;

    public static void Run(
        Trainer trainer,
        (int[] input, int[] target)[] pairs,
        Vocabulary vocab,
        int epochs,
        int reportEvery = 200)
    {
        Console.Clear();
        Console.CursorVisible = false;

        Console.WriteLine($"Training — {epochs} epochs");
        Console.WriteLine(new string('═', 48));
        int panelTop = Console.CursorTop;

        // Reserve lines so later SetCursorPosition doesn't scroll
        int panelHeight = 3 + 6 + 2 + pairs.Length + 2;
        for (int i = 0; i < panelHeight; i++) Console.WriteLine();
        int panelBottom = Console.CursorTop;

        var history      = new List<float>();
        float initLoss   = 0f;
        bool skipRequested = false;
        float finalLoss  = 0f;

        try
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                if (!skipRequested && Console.KeyAvailable)
                    if (Console.ReadKey(intercept: true).Key == ConsoleKey.Q)
                        skipRequested = true;

                var (avgLoss, perPairLosses) = trainer.TrainStep(pairs);
                finalLoss = avgLoss;

                if (epoch == 0) initLoss = avgLoss;

                if (epoch % reportEvery == 0 || epoch == epochs - 1)
                {
                    history.Add(avgLoss);
                    DrawPanel(panelTop, epoch + 1, epochs, avgLoss, initLoss,
                              history, perPairLosses, pairs, vocab);
                }

                if (skipRequested) break;
            }
        }
        finally
        {
            int safeRow = Math.Min(panelBottom, Console.BufferHeight - 1);
            Console.SetCursorPosition(0, safeRow);
            Console.CursorVisible = true;
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"✓ Done — loss {finalLoss:F4}");
        Console.ResetColor();
        Console.WriteLine();
    }

    private static void DrawPanel(
        int panelTop, int epoch, int epochs, float avgLoss, float initLoss,
        List<float> history, float[] perPairLosses,
        (int[] input, int[] target)[] pairs, Vocabulary vocab)
    {
        Console.SetCursorPosition(0, panelTop);

        // Progress line
        float pct    = (float)epoch / epochs;
        int   filled = (int)(pct * BarWidth);
        string bar   = new string('█', filled) + new string('░', BarWidth - filled);
        Overwrite($"Epoch {epoch,6} / {epochs}  [{bar}] {pct * 100,4:F0}%");
        Overwrite("");

        // Loss curve — last 6 checkpoints
        Overwrite("Loss curve (last 6 checkpoints):");
        float maxLoss = initLoss > 0f ? initLoss : 1f;
        int   start   = Math.Max(0, history.Count - 6);
        for (int i = start; i < history.Count; i++)
        {
            float l = history[i];
            int   w = Math.Clamp((int)(l / maxLoss * BarWidth), 0, BarWidth);
            string marker = i == history.Count - 1 ? "  ←" : "   ";
            Console.ForegroundColor = l > maxLoss * 0.5f ? ConsoleColor.Red
                                    : l > maxLoss * 0.1f ? ConsoleColor.Yellow
                                    : ConsoleColor.Green;
            Overwrite($"  {l:F4} {new string('█', w)}{new string('░', BarWidth - w)}{marker}");
            Console.ResetColor();
        }
        // Blank lines for unused history slots
        for (int i = history.Count - start; i < 6; i++) Overwrite("");

        Overwrite("");
        Overwrite("Per-pair losses:");
        float avg     = perPairLosses.Average();
        float maxPair = Math.Max(perPairLosses.Max(), 1e-6f);

        for (int i = 0; i < pairs.Length; i++)
        {
            string label = vocab.Decode(pairs[i].input);
            if (label.Length > PairLabelW) label = label[..(PairLabelW - 3)] + "...";
            label = ("\"" + label + "\"").PadRight(PairLabelW + 2);
            int    w    = Math.Clamp((int)(perPairLosses[i] / maxPair * 8), 0, 8);
            string flag = perPairLosses[i] > avg ? " ▲" : "  ";
            Console.ForegroundColor = perPairLosses[i] > avg
                ? ConsoleColor.Yellow : ConsoleColor.DarkGray;
            Overwrite($"  {label} {new string('█', w)}{new string('░', 8 - w)} {perPairLosses[i]:F4}{flag}");
            Console.ResetColor();
        }

        Console.ResetColor();
        Overwrite("");
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Overwrite("Press Q to skip remaining epochs");
        Console.ResetColor();
    }

    // Write a line that pads to window width to erase any previous longer content.
    private static void Overwrite(string text)
    {
        int w = Console.WindowWidth > 0 ? Console.WindowWidth - 1 : 79;
        Console.WriteLine(text.Length < w ? text.PadRight(w) : text[..w]);
    }
}
