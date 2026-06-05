namespace YodaTransformer;

public static class VerboseInference
{
    public static void Run(TransformerModel model, Vocabulary vocab, string sentence)
    {
        int[] tokens;
        try { tokens = vocab.Encode(sentence.Trim()); }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"  (unknown word — {ex.Message})");
            Console.WriteLine();
            return;
        }

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write("You: ");
        Console.ResetColor();
        Console.WriteLine(sentence);
        Console.WriteLine(new string('─', 56));

        // ─── ① Tokenise ───────────────────────────────────────
        Section("① Tokenise");
        Dim("  Each word is mapped to an integer ID from the vocabulary.");
        Dim("  Special tokens <bos> (start) and <eos> (end) are added automatically.");

        // Words row
        Console.Write("  ");
        foreach (int t in tokens)
            Console.Write($"[ {vocab.WordAt(t),-7}]  ");
        Console.WriteLine();
        // IDs row
        Console.Write("  ");
        foreach (int t in tokens)
            Console.Write($"  {t,-9} ");
        Console.WriteLine();
        Console.WriteLine();

        // ─── Run forward pass (populates all caches) ──────────
        float[][] logits = model.Forward(tokens);
        float[][] embOut = model.Embedding.LastOutput;

        // ─── ② Embed + Positional Encoding ────────────────────
        Section($"② Embed + Positional Encoding  ({embOut[0].Length}-dim)");
        Dim("  Each token ID is looked up in a learned embedding table.");
        Dim("  A fixed sinusoidal positional encoding is added so the model");
        Dim("  knows the order of tokens.");

        for (int p = 0; p < tokens.Length; p++)
        {
            string label = $"pos {p} {vocab.WordAt(tokens[p])}:".PadRight(16);
            string vals  = string.Join(" ", embOut[p].Take(8).Select(v => $"{v,7:F3}"));
            Console.WriteLine($"  {label} [ {vals} ... ]");
        }
        Console.WriteLine();

        // ─── ③ Attention ──────────────────────────────────────
        float[][] attnW = model.Block.Attention.Weights;
        Section("③ Attention  (single head)");
        Dim("  Each head learns to attend to different token relationships.");
        Dim("  The weight matrix shows how much each position (row) attends");
        Dim("  to each other position (col) after softmax normalisation.");

        // Column header
        Console.Write("            ");
        foreach (int t in tokens)
            Console.Write($"{vocab.WordAt(t),8}");
        Console.WriteLine();
        // Rows
        for (int r = 0; r < tokens.Length; r++)
        {
            Console.Write($"  {vocab.WordAt(tokens[r]),-10}");
            Console.Write("[ ");
            float rowMax = attnW[r].Max();
            for (int c = 0; c < tokens.Length; c++)
            {
                if (attnW[r][c] == rowMax) Console.ForegroundColor = ConsoleColor.Green;
                Console.Write($"{attnW[r][c],7:F3} ");
                Console.ResetColor();
            }
            Console.WriteLine("]");
        }
        Console.WriteLine();

        // ─── ④ Output logits → predictions ────────────────────
        Section("④ Output logits → predictions");
        Dim("  The transformer's output vectors are projected to vocabulary size");
        Dim("  via Wout. The highest logit at each position becomes the output token.");

        for (int p = 1; p < tokens.Length - 1; p++) // skip BOS (p=0) and EOS (p=last)
        {
            Console.WriteLine($"  pos {p} ({vocab.WordAt(tokens[p])}) — top tokens:");
            PrintLogitBars(logits[p], vocab, topN: 3);
        }

        Console.WriteLine(new string('─', 56));
        int[] predicted = logits.Select(row => row
            .Select((v, i) => (v, i))
            .OrderByDescending(x => x.v)
            .First().i).ToArray();
        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write("Yoda: ");
        Console.ResetColor();
        Console.WriteLine(vocab.Decode(predicted));
        Console.WriteLine();
    }

    private static void Section(string text)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine(text);
        Console.ResetColor();
    }

    private static void Dim(string text)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine(text);
        Console.ResetColor();
    }

    private static void PrintLogitBars(float[] logits, Vocabulary vocab, int topN)
    {
        var ranked = Enumerable.Range(0, logits.Length)
                               .OrderByDescending(i => logits[i])
                               .Take(topN)
                               .ToArray();
        float maxLogit = logits[ranked[0]];
        float minLogit = logits[ranked[^1]];
        float span     = maxLogit - minLogit;

        foreach (int idx in ranked)
        {
            string word = vocab.WordAt(idx).PadRight(10);
            int    w    = span > 1e-6f
                          ? Math.Clamp((int)((logits[idx] - minLogit) / span * 20), 0, 20)
                          : (idx == ranked[0] ? 20 : 0);
            string bar      = new string('█', w) + new string('░', 20 - w);
            string selected = idx == ranked[0] ? "  ✓" : "";
            if (idx == ranked[0]) Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"    {word} {bar}  {logits[idx],6:F2}{selected}");
            Console.ResetColor();
        }
    }
}
