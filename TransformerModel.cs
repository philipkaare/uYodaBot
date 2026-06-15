namespace YodaTransformer;

public class TransformerModel
{
    private EmbeddingLayer _embedding;
    private TransformerBlock _block;
    public  float[][] Wout;
    private float[][] _embOut = null!;
    private float[][] _blockOut = null!;

    public EmbeddingLayer Embedding => _embedding;
    public TransformerBlock Block => _block;

    public TransformerModel(int vocabSize, int dModel, int dHead, int dFF, int maxSeqLen, Random rng)
    {
        _embedding = new EmbeddingLayer(vocabSize, dModel, maxSeqLen, rng);
        _block = new TransformerBlock(dModel, dHead, dFF, rng);
        Wout = MathOps.RandomMatrix(dModel, vocabSize, rng, scale: 0.02f);
    }

    public float[][] Forward(int[] tokens)
    {
        _embOut = _embedding.Forward(tokens);
        _blockOut = _block.Forward(_embOut);
        int seqLen = tokens.Length;
        float[][] logits = new float[seqLen][];
        for (int p = 0; p < seqLen; p++)
            logits[p] = MathOps.VecMatMul(_blockOut[p], Wout);
        return logits;
    }

    public (float[][] dWout,
            float[][] dWq, float[][] dWk, float[][] dWv, float[][] dWo,
            float[][] dW1, float[] db1, float[][] dW2, float[] db2,
            float[][] dTokenWeights)
        Backward(int[] tokens, float[][] dLogits)
    {
        int seqLen = tokens.Length;
        int dModel = _blockOut[0].Length;
        int vocabSize = Wout[0].Length;

        float[][] dWout = new float[dModel][];
        for (int r = 0; r < dModel; r++)
            dWout[r] = new float[vocabSize];

        float[][] dBlockOut = new float[seqLen][];
        for (int p = 0; p < seqLen; p++)
        {
            for (int r = 0; r < dModel; r++)
                for (int c = 0; c < vocabSize; c++)
                    dWout[r][c] += _blockOut[p][r] * dLogits[p][c];

            dBlockOut[p] = MathOps.VecMatMul(dLogits[p], MathOps.Transpose(Wout));
        }

        var (dEmbOut, dWq, dWk, dWv, dWo, dW1, db1, dW2, db2) = _block.Backward(dBlockOut);

        _embedding.Backward(tokens, dEmbOut);
        float[][] dTokenWeights = _embedding.TokenWeights;

        return (dWout, dWq, dWk, dWv, dWo, dW1, db1, dW2, db2, dTokenWeights);
    }

    public int[] Predict(int[] tokens)
    {
        float[][] logits = Forward(tokens);
        int seqLen = tokens.Length;
        int[] result = new int[seqLen];
        for (int p = 0; p < seqLen; p++)
        {
            int best = 0;
            for (int i = 1; i < logits[p].Length; i++)
                if (logits[p][i] > logits[p][best]) best = i;
            result[p] = best;
        }
        return result;
    }
}
