namespace YodaTransformer;

public class AttentionHead
{
    private int _dHead;

    public float[][] Wq;
    public float[][] Wk;
    public float[][] Wv;
    public float[][] Wo;

    private float[][] _x;
    private float[][] _q;
    private float[][] _k;
    private float[][] _v;
    private float[][] _scores;
    private float[][] _weights;
    private float[][] _context;

    public float[][] Weights => _weights;
    public float[][] Q       => _q;
    public float[][] K       => _k;
    public float[][] V       => _v;

    public AttentionHead(int dModel, int dHead, Random rng)
    {
        _dHead = dHead;
        Wq = MathOps.RandomMatrix(dModel, dHead, rng, scale: 0.02f);
        Wk = MathOps.RandomMatrix(dModel, dHead, rng, scale: 0.02f);
        Wv = MathOps.RandomMatrix(dModel, dHead, rng, scale: 0.02f);
        Wo = MathOps.RandomMatrix(dHead, dModel, rng, scale: 0.02f);
    }

    public float[][] Forward(float[][] x)
    {
        _x = x;
        _q = MathOps.MatMul(x, Wq);
        _k = MathOps.MatMul(x, Wk);
        _v = MathOps.MatMul(x, Wv);
        _scores = MathOps.ScalarMul(MathOps.MatMul(_q, MathOps.Transpose(_k)), 1f / MathF.Sqrt(_dHead));
        _weights = MathOps.SoftmaxRows(_scores);
        _context = MathOps.MatMul(_weights, _v);
        return MathOps.MatMul(_context, Wo);
    }

    public (float[][] dX, float[][] dWq, float[][] dWk, float[][] dWv, float[][] dWo) Backward(float[][] dOut)
    {
        float[][] dContext = MathOps.MatMul(dOut, MathOps.Transpose(Wo));
        float[][] dWo = MathOps.MatMul(MathOps.Transpose(_context), dOut);

        float[][] dWeights = MathOps.MatMul(dContext, MathOps.Transpose(_v));
        float[][] dScores = SoftmaxBackward(_weights, dWeights);
        float[][] dV = MathOps.MatMul(MathOps.Transpose(_weights), dContext);

        float[][] dScoresRaw = MathOps.ScalarMul(dScores, 1f / MathF.Sqrt(_dHead));

        float[][] dQ = MathOps.MatMul(dScoresRaw, _k);
        float[][] dK = MathOps.MatMul(MathOps.Transpose(dScoresRaw), _q);

        float[][] dWq = MathOps.MatMul(MathOps.Transpose(_x), dQ);
        float[][] dWk = MathOps.MatMul(MathOps.Transpose(_x), dK);
        float[][] dWv = MathOps.MatMul(MathOps.Transpose(_x), dV);

        float[][] dX = MathOps.AddMatrices(
            MathOps.AddMatrices(
                MathOps.MatMul(dQ, MathOps.Transpose(Wq)),
                MathOps.MatMul(dK, MathOps.Transpose(Wk))),
            MathOps.MatMul(dV, MathOps.Transpose(Wv)));

        return (dX, dWq, dWk, dWv, dWo);
    }

    private static float[][] SoftmaxBackward(float[][] weights, float[][] dWeights)
    {
        int seqLen = weights.Length;
        float[][] dScores = new float[seqLen][];
        for (int i = 0; i < seqLen; i++)
        {
            int cols = weights[i].Length;
            dScores[i] = new float[cols];
            float dot = 0f;
            for (int j = 0; j < cols; j++)
                dot += weights[i][j] * dWeights[i][j];
            for (int j = 0; j < cols; j++)
                dScores[i][j] = weights[i][j] * (dWeights[i][j] - dot);
        }
        return dScores;
    }
}
