namespace YodaTransformer;

public class TransformerBlock
{
    private AttentionHead _attention;
    private FeedForward _ffn;

    private float[][] _x = null!;
    private float[][] _xNorm1 = null!;
    private float[][] _attnOut = null!;
    private float[][] _x1 = null!;
    private float[][] _xNorm2 = null!;
    private float[][] _ffnOut = null!;

    public AttentionHead Attention => _attention;
    public FeedForward   Ffn       => _ffn;
    public float[][]     FfnOut    => _ffnOut;

    public TransformerBlock(int dModel, int dHead, int dFF, Random rng)
    {
        _attention = new AttentionHead(dModel, dHead, rng);
        _ffn = new FeedForward(dModel, dFF, rng);
    }

    public float[][] Forward(float[][] x)
    {
        _x = x;
        _xNorm1 = MathOps.LayerNormRows(x);
        _attnOut = _attention.Forward(_xNorm1);
        _x1 = MathOps.AddMatrices(x, _attnOut);
        _xNorm2 = MathOps.LayerNormRows(_x1);
        _ffnOut = _ffn.Forward(_xNorm2);
        return MathOps.AddMatrices(_x1, _ffnOut);
    }

    public (float[][] dX,
            float[][] dWq, float[][] dWk, float[][] dWv, float[][] dWo,
            float[][] dW1, float[] db1, float[][] dW2, float[] db2)
        Backward(float[][] dOut)
    {
        float[][] dX1_from_residual = dOut;
        float[][] dFfnOut = dOut;

        var (dXNorm2, dW1, db1, dW2, db2) = _ffn.Backward(dFfnOut);

        float[][] dX1_from_norm2 = LayerNormBackwardRows(_x1, dXNorm2);

        float[][] dX1 = MathOps.AddMatrices(dX1_from_residual, dX1_from_norm2);

        float[][] dX_from_residual = dX1;
        float[][] dAttnOut = dX1;

        var (dXNorm1, dWq, dWk, dWv, dWo) = _attention.Backward(dAttnOut);

        float[][] dX_from_norm1 = LayerNormBackwardRows(_x, dXNorm1);

        float[][] dX = MathOps.AddMatrices(dX_from_residual, dX_from_norm1);

        return (dX, dWq, dWk, dWv, dWo, dW1, db1, dW2, db2);
    }

    private static float[][] LayerNormBackwardRows(float[][] x, float[][] dOut)
    {
        int seqLen = x.Length;
        float[][] dx = new float[seqLen][];

        for (int i = 0; i < seqLen; i++)
        {
            float[] xi = x[i];
            float[] dOuti = dOut[i];
            int N = xi.Length;

            float mean = 0f;
            for (int j = 0; j < N; j++)
                mean += xi[j];
            mean /= N;

            float variance = 0f;
            for (int j = 0; j < N; j++)
            {
                float diff = xi[j] - mean;
                variance += diff * diff;
            }
            variance /= N;

            float std = MathF.Sqrt(variance + 1e-6f);

            float[] dx_hat = dOuti;

            float dvar = 0f;
            for (int j = 0; j < N; j++)
                dvar += dx_hat[j] * (xi[j] - mean) * -0.5f * MathF.Pow(variance + 1e-6f, -1.5f);

            float dmean_part1 = 0f;
            for (int j = 0; j < N; j++)
                dmean_part1 += dx_hat[j] * (-1f / std);

            float sum_diff = 0f;
            for (int j = 0; j < N; j++)
                sum_diff += xi[j] - mean;

            float dmean = dmean_part1 + dvar * (-2f / N) * sum_diff;

            dx[i] = new float[N];
            for (int j = 0; j < N; j++)
                dx[i][j] = dx_hat[j] / std + dvar * 2f * (xi[j] - mean) / N + dmean / N;
        }

        return dx;
    }
}
