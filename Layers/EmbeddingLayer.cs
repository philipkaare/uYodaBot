namespace YodaTransformer;

public class EmbeddingLayer
{
    public float[][] TokenWeights;
    private float[][] _posEncoding;
    private int _dModel;
    private float[][] _lastOutput = null!;
    public  float[][] LastOutput             => _lastOutput;
    public  float[]   GetPosEncoding(int pos) => _posEncoding[pos];

    public EmbeddingLayer(int vocabSize, int dModel, int maxSeqLen, Random rng)
    {
        _dModel = dModel;
        TokenWeights = MathOps.RandomMatrix(vocabSize, dModel, rng, scale: 0.01f);

        _posEncoding = new float[maxSeqLen][];
        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            _posEncoding[pos] = new float[dModel];
            for (int i = 0; i < dModel; i++)
            {
                float angle = pos / MathF.Pow(10000f, (float)(i % 2 == 0 ? i : i - 1) / dModel);
                _posEncoding[pos][i] = i % 2 == 0 ? MathF.Sin(angle) : MathF.Cos(angle);
            }
        }
    }

    public float[][] Forward(int[] tokens)
    {
        float[][] output = new float[tokens.Length][];
        for (int p = 0; p < tokens.Length; p++)
            output[p] = MathOps.AddVectors(TokenWeights[tokens[p]], _posEncoding[p]);
        _lastOutput = output;
        return output;
    }

    public void Backward(int[] tokens, float[][] grad)
    {
        for (int p = 0; p < tokens.Length; p++)
        {
            int t = tokens[p];
            for (int i = 0; i < _dModel; i++)
                TokenWeights[t][i] += grad[p][i];
        }
    }
}
