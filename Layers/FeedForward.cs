namespace YodaTransformer;

public class FeedForward
{
    // Weight fields (public for trainer access)
    public float[][] W1;   // [dModel × dFF]
    public float[]   b1;   // [dFF]
    public float[][] W2;   // [dFF × dModel]
    public float[]   b2;   // [dModel]

    // Cached activations
    private float[][] _x;      // input [seqLen × dModel]
    private float[][] _h;      // pre-ReLU hidden [seqLen × dFF]
    private float[][] _hRelu;  // post-ReLU hidden [seqLen × dFF]
    public  float[][] HRelu => _hRelu;

    public FeedForward(int dModel, int dFF, Random rng)
    {
        W1 = MathOps.RandomMatrix(dModel, dFF, rng, scale: 0.02f);
        b1 = MathOps.RandomVector(dFF, rng, scale: 0.01f);
        W2 = MathOps.RandomMatrix(dFF, dModel, rng, scale: 0.02f);
        b2 = MathOps.RandomVector(dModel, rng, scale: 0.01f);
    }

    public float[][] Forward(float[][] x)
    {
        int seqLen = x.Length;
        int dModel = x[0].Length;
        int dFF = W1[0].Length;

        _x = x;
        _h = new float[seqLen][];
        _hRelu = new float[seqLen][];
        float[][] output = new float[seqLen][];

        for (int p = 0; p < seqLen; p++)
        {
            // h[p] = AddVectors(VecMatMul(x[p], W1), b1) → [dFF]
            _h[p] = MathOps.AddVectors(MathOps.VecMatMul(x[p], W1), b1);

            // hRelu[p] = ReLU(h[p]) → [dFF]
            _hRelu[p] = MathOps.ReLU(_h[p]);

            // out[p] = AddVectors(VecMatMul(hRelu[p], W2), b2) → [dModel]
            output[p] = MathOps.AddVectors(MathOps.VecMatMul(_hRelu[p], W2), b2);
        }

        return output;
    }

    public (float[][] dX, float[][] dW1, float[] db1, float[][] dW2, float[] db2) Backward(float[][] dOut)
    {
        int seqLen = dOut.Length;
        int dModel = dOut[0].Length;
        int dFF = _h[0].Length;

        // Initialize gradients
        float[][] dX = new float[seqLen][];
        for (int i = 0; i < seqLen; i++)
            dX[i] = new float[dModel];

        float[][] dW1 = new float[dModel][];
        for (int i = 0; i < dModel; i++)
            dW1[i] = new float[dFF];

        float[] db1 = new float[dFF];

        float[][] dW2 = new float[dFF][];
        for (int i = 0; i < dFF; i++)
            dW2[i] = new float[dModel];

        float[] db2 = new float[dModel];

        // Backward pass
        float[][] dH = new float[seqLen][];
        for (int i = 0; i < seqLen; i++)
            dH[i] = new float[dFF];

        float[][] dHRelu = new float[seqLen][];
        for (int i = 0; i < seqLen; i++)
            dHRelu[i] = new float[dFF];

        for (int p = 0; p < seqLen; p++)
        {
            // db2 += dOut[p]
            for (int j = 0; j < dModel; j++)
                db2[j] += dOut[p][j];

            // dHRelu[p] = VecMatMul(dOut[p], Transpose(W2)) → [dFF]
            dHRelu[p] = MathOps.VecMatMul(dOut[p], MathOps.Transpose(W2));

            // dW2 contribution: outer product of _hRelu[p] and dOut[p] → accumulate into dW2 [dFF × dModel]
            for (int r = 0; r < dFF; r++)
            {
                for (int c = 0; c < dModel; c++)
                {
                    dW2[r][c] += _hRelu[p][r] * dOut[p][c];
                }
            }

            // ReLU backward: dH[p][j] = dHRelu[p][j] * (_h[p][j] > 0 ? 1f : 0f)
            for (int j = 0; j < dFF; j++)
                dH[p][j] = dHRelu[p][j] * (_h[p][j] > 0f ? 1f : 0f);

            // db1 += dH[p]
            for (int j = 0; j < dFF; j++)
                db1[j] += dH[p][j];

            // dX[p] = VecMatMul(dH[p], Transpose(W1)) → [dModel]
            dX[p] = MathOps.VecMatMul(dH[p], MathOps.Transpose(W1));

            // dW1 contribution: outer product of _x[p] and dH[p] → accumulate into dW1 [dModel × dFF]
            for (int r = 0; r < dModel; r++)
            {
                for (int c = 0; c < dFF; c++)
                {
                    dW1[r][c] += _x[p][r] * dH[p][c];
                }
            }
        }

        return (dX, dW1, db1, dW2, db2);
    }
}
