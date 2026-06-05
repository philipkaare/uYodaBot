namespace YodaTransformer;

public static class ModelSerializer
{
    private static readonly byte[] Magic   = { (byte)'Y', (byte)'O', (byte)'D', (byte)'A' };
    private const byte Version = 1;

    public static void Save(TransformerModel model, int epochs, string path)
    {
        using var w = new BinaryWriter(File.Open(path, FileMode.Create));
        w.Write(Magic);
        w.Write(Version);
        w.Write(epochs);

        WriteMatrix(w, model.Embedding.TokenWeights);
        WriteMatrix(w, model.Wout);

        WriteMatrix(w, model.Block.Attention.Wq);
        WriteMatrix(w, model.Block.Attention.Wk);
        WriteMatrix(w, model.Block.Attention.Wv);
        WriteMatrix(w, model.Block.Attention.Wo);

        WriteMatrix(w, model.Block.Ffn.W1);
        WriteVec   (w, model.Block.Ffn.b1);
        WriteMatrix(w, model.Block.Ffn.W2);
        WriteVec   (w, model.Block.Ffn.b2);
    }

    public static bool TryLoad(TransformerModel model, out int epochs, string path)
    {
        epochs = 0;
        if (!File.Exists(path)) return false;
        try
        {
            using var r = new BinaryReader(File.OpenRead(path));
            if (!r.ReadBytes(4).SequenceEqual(Magic)) return false;
            if (r.ReadByte() != Version) return false;
            epochs = r.ReadInt32();

            ReadIntoMatrix(r, model.Embedding.TokenWeights);
            ReadIntoMatrix(r, model.Wout);

            ReadIntoMatrix(r, model.Block.Attention.Wq);
            ReadIntoMatrix(r, model.Block.Attention.Wk);
            ReadIntoMatrix(r, model.Block.Attention.Wv);
            ReadIntoMatrix(r, model.Block.Attention.Wo);

            ReadIntoMatrix(r, model.Block.Ffn.W1);
            ReadIntoVec   (r, model.Block.Ffn.b1);
            ReadIntoMatrix(r, model.Block.Ffn.W2);
            ReadIntoVec   (r, model.Block.Ffn.b2);

            return true;
        }
        catch { return false; }
    }

    private static void WriteMatrix(BinaryWriter w, float[][] m)
    {
        w.Write(m.Length);
        w.Write(m[0].Length);
        foreach (var row in m)
            foreach (var v in row)
                w.Write(v);
    }

    private static void WriteVec(BinaryWriter w, float[] v)
    {
        w.Write(1);
        w.Write(v.Length);
        foreach (var f in v) w.Write(f);
    }

    private static void ReadIntoMatrix(BinaryReader r, float[][] m)
    {
        int rows = r.ReadInt32();
        int cols = r.ReadInt32();
        if (rows != m.Length || cols != m[0].Length)
            throw new InvalidOperationException(
                $"Dimension mismatch: file [{rows}×{cols}] vs model [{m.Length}×{m[0].Length}]");
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = r.ReadSingle();
    }

    private static void ReadIntoVec(BinaryReader r, float[] v)
    {
        r.ReadInt32(); // rows = 1, skip
        int cols = r.ReadInt32();
        if (cols != v.Length)
            throw new InvalidOperationException(
                $"Vector length mismatch: file [{cols}] vs model [{v.Length}]");
        for (int j = 0; j < cols; j++)
            v[j] = r.ReadSingle();
    }
}
