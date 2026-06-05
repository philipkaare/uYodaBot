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

        Console.Write("  ");
        foreach (int t in tokens)
            Console.Write($"[ {vocab.WordAt(t),-7}]  ");
        Console.WriteLine();
        Console.Write("  ");
        foreach (int t in tokens)
            Console.Write($"  {t,-9} ");
        Console.WriteLine();
        Console.WriteLine();

        // ─── Run forward pass (populates all caches) ──────────
        float[][] logits = model.Forward(tokens);
        float[][] embOut = model.Embedding.LastOutput;

        // ─── ② Embed + Positional Encoding ────────────────────
        int dModel = embOut[0].Length;
        Section($"② Embed + Positional Encoding  ({dModel}-dim)");
        Dim("  Token embedding: each word ID maps to a row in a learned weight matrix.");
        Dim("  The values have no fixed human meaning — the model assigns whatever");
        Dim("  numbers minimise loss during training. Similar words end up with");
        Dim("  similar vectors. The embedding for the same word is always identical.");
        Dim("");
        Dim("  Positional encoding: a fixed sin/cos pattern added to each vector so");
        Dim("  the model knows word order. Even dims use sin, odd dims use cos, each");
        Dim("  at a different frequency. Position 0 always adds the same offset.");
        Console.WriteLine();

        int show = Math.Min(5, dModel);
        for (int p = 0; p < tokens.Length; p++)
        {
            float[] tokEmb = model.Embedding.TokenWeights[tokens[p]];
            float[] posEnc = model.Embedding.GetPosEncoding(p);

            string embStr = string.Join(" ", tokEmb.Take(show).Select(v => $"{v,7:F3}"));
            string posStr = string.Join(" ", posEnc.Take(show).Select(v => $"{v,7:F3}"));
            string totStr = string.Join(" ", embOut[p].Take(show).Select(v => $"{v,7:F3}"));

            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"  pos {p} ({vocab.WordAt(tokens[p])})");
            Console.ResetColor();
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"    embed [ {embStr} ... ]  (word features)");
            Console.WriteLine($"    + pos [ {posStr} ... ]  (position signal)");
            Console.ResetColor();
            Console.WriteLine($"    total [ {totStr} ... ]");
        }
        Console.WriteLine();

        // ─── ③ Attention ──────────────────────────────────────
        float[][] attnW = model.Block.Attention.Weights;
        Section("③ Attention  (single head)");
        Dim("  Each head learns to attend to different token relationships.");
        Dim("  The weight matrix shows how much each position (row) attends");
        Dim("  to each other position (col) after softmax normalisation.");

        Console.Write("            ");
        foreach (int t in tokens)
            Console.Write($"{vocab.WordAt(t),8}");
        Console.WriteLine();
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

        // ─── ④ Feed-Forward Network ───────────────────────────
        float[][] ffnOut = model.Block.FfnOut;
        float[][] hRelu  = model.Block.Ffn.HRelu;
        int       dFF    = hRelu[0].Length;

        Section($"④ Feed-Forward Network  ({dFF} hidden neurons, ReLU)");
        Dim("  After attention, each position independently passes through a tiny");
        Dim("  2-layer perceptron: x -> W1 -> ReLU -> W2 -> output.");
        Dim($"  W1 projects from {dModel} dims up to {dFF} dims (the hidden layer).");
        Dim("  ReLU zeroes any negative pre-activations, leaving only positive signals.");
        Dim($"  W2 projects back down to {dModel} dims and the result is added to the");
        Dim("  residual stream (skip connection — the original vector is preserved).");
        Dim("  This is where the model stores factual and structural patterns.");
        Console.WriteLine();

        for (int p = 0; p < tokens.Length; p++)
        {
            int    active = hRelu[p].Count(v => v > 0f);
            string fires  = new string(hRelu[p].Select(v => v > 0f ? '█' : '░').ToArray());
            string posLabel = $"pos {p} ({vocab.WordAt(tokens[p])})".PadRight(14);

            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"  {posLabel}  ");
            Console.ResetColor();
            Console.ForegroundColor = active > dFF / 2 ? ConsoleColor.Green : ConsoleColor.Yellow;
            Console.Write(fires);
            Console.ResetColor();
            Console.WriteLine($"  {active}/{dFF} neurons active");

            string outStr = string.Join(" ", ffnOut[p].Take(show).Select(v => $"{v,7:F3}"));
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  {"ffn output:",-18}[ {outStr} ... ]");
            Console.ResetColor();
        }
        Console.WriteLine();

        // ─── ⑤ Output logits → predictions ────────────────────
        Section("⑤ Output logits  (predictions)");
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
