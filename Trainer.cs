namespace YodaTransformer;

public class Trainer
{
    private TransformerModel _model;
    private float _lr;
    // Stores the true embedding weight values separately from the gradient accumulator.
    private float[][] _embWeightsCopy;

    public Trainer(TransformerModel model, float learningRate = 0.01f)
    {
        _model = model;
        _lr = learningRate;

        // Make a deep copy of the initial embedding weights.
        float[][] src = _model.Embedding.TokenWeights;
        int vocabSize = src.Length;
        int dModel = src[0].Length;
        _embWeightsCopy = new float[vocabSize][];
        for (int i = 0; i < vocabSize; i++)
        {
            _embWeightsCopy[i] = new float[dModel];
            Array.Copy(src[i], _embWeightsCopy[i], dModel);
        }
    }

    // Numerically-stable cross-entropy loss averaged over all sequence positions.
    private float CrossEntropyLoss(float[][] logits, int[] targets)
    {
        int seqLen = logits.Length;
        float totalLoss = 0f;

        for (int p = 0; p < seqLen; p++)
        {
            float[] row = logits[p];
            int vocabSize = row.Length;

            float maxVal = row[0];
            for (int j = 1; j < vocabSize; j++)
                if (row[j] > maxVal) maxVal = row[j];

            float sumExp = 0f;
            for (int j = 0; j < vocabSize; j++)
                sumExp += MathF.Exp(row[j] - maxVal);

            float logSumExp = MathF.Log(sumExp);

            totalLoss += -(row[targets[p]] - maxVal - logSumExp);
        }

        return totalLoss / seqLen;
    }

    // Cross-entropy gradient: softmax(logits[p]) - one_hot(targets[p]), averaged over positions.
    private float[][] CrossEntropyGrad(float[][] logits, int[] targets)
    {
        int seqLen = logits.Length;
        int vocabSize = logits[0].Length;
        float[][] dLogits = new float[seqLen][];

        for (int p = 0; p < seqLen; p++)
        {
            float[] sm = MathOps.Softmax(logits[p]);
            sm[targets[p]] -= 1f;
            // Divide by seqLen to average over positions.
            for (int j = 0; j < vocabSize; j++)
                sm[j] /= seqLen;
            dLogits[p] = sm;
        }

        return dLogits;
    }

    // Flatten dW, compute clip scale, then apply clipped SGD update to W.
    private static void ClipAndUpdate(float[][] W, float[][] dW, float lr, float maxNorm)
    {
        int rows = dW.Length;
        int cols = dW[0].Length;

        // Flatten gradient into a 1-D array for norm computation.
        float[] flat = new float[rows * cols];
        int idx = 0;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                flat[idx++] = dW[r][c];

        float scale = MathOps.ClipNorm(flat, maxNorm);

        // Apply scaled update.
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                W[r][c] -= lr * dW[r][c] * scale;
    }

    // Flatten db, compute clip scale, then apply clipped SGD update to b.
    private static void ClipAndUpdateVec(float[] b, float[] db, float lr, float maxNorm)
    {
        // Copy gradient into a flat array before norm computation (mirrors ClipAndUpdate pattern).
        float[] flat = new float[db.Length];
        for (int i = 0; i < db.Length; i++)
            flat[i] = db[i];

        float scale = MathOps.ClipNorm(flat, maxNorm);
        for (int i = 0; i < b.Length; i++)
            b[i] -= lr * db[i] * scale;
    }

    // Run one training step over the provided pairs and return the average loss and per-pair losses.
    public (float avgLoss, float[] perPairLosses) TrainStep((int[] input, int[] target)[] pairs)
    {
        float totalLoss = 0f;
        float[] perPairLosses = new float[pairs.Length];

        for (int pairIdx = 0; pairIdx < pairs.Length; pairIdx++)
        {
            var pair = pairs[pairIdx];

            // ── 1. Restore true embedding weights before Forward.
            float[][] tw = _model.Embedding.TokenWeights;
            int vocabSize = tw.Length;
            int dModel = tw[0].Length;
            for (int i = 0; i < vocabSize; i++)
                Array.Copy(_embWeightsCopy[i], tw[i], dModel);

            // ── 2. Forward pass.
            float[][] logits = _model.Forward(pair.input);

            // ── 3. Compute loss.
            float pairLoss = CrossEntropyLoss(logits, pair.target);
            totalLoss += pairLoss;
            perPairLosses[pairIdx] = pairLoss;

            // ── 4. Compute output gradient.
            float[][] dLogits = CrossEntropyGrad(logits, pair.target);

            // ── 5. Zero out TokenWeights so Backward uses it as gradient accumulator.
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < dModel; j++)
                    tw[i][j] = 0f;

            // ── 6. Backward pass.
            var (dWout, dWq, dWk, dWv, dWo, dW1, db1, dW2, db2, dTokenWeights) =
                _model.Backward(pair.input, dLogits);

            // ── 7. Apply clipped SGD updates.
            ClipAndUpdate(_model.Wout, dWout, _lr, 1.0f);
            ClipAndUpdate(_model.Block.Attention.Wq, dWq, _lr, 1.0f);
            ClipAndUpdate(_model.Block.Attention.Wk, dWk, _lr, 1.0f);
            ClipAndUpdate(_model.Block.Attention.Wv, dWv, _lr, 1.0f);
            ClipAndUpdate(_model.Block.Attention.Wo, dWo, _lr, 1.0f);
            ClipAndUpdate(_model.Block.Ffn.W1, dW1, _lr, 1.0f);
            ClipAndUpdateVec(_model.Block.Ffn.b1, db1, _lr, 1.0f);
            ClipAndUpdate(_model.Block.Ffn.W2, dW2, _lr, 1.0f);
            ClipAndUpdateVec(_model.Block.Ffn.b2, db2, _lr, 1.0f);

            // ── 8. Apply embedding gradient update via copy.
            float[] flatEmb = new float[vocabSize * dModel];
            int idx = 0;
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < dModel; j++)
                    flatEmb[idx++] = dTokenWeights[i][j];
            float embScale = MathOps.ClipNorm(flatEmb, 1.0f);
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < dModel; j++)
                    _embWeightsCopy[i][j] -= _lr * dTokenWeights[i][j] * embScale;
        }

        // ── 9. Restore trained embedding weights for inference.
        float[][] finalTw = _model.Embedding.TokenWeights;
        for (int i = 0; i < finalTw.Length; i++)
            Array.Copy(_embWeightsCopy[i], finalTw[i], _embWeightsCopy[i].Length);

        return (totalLoss / pairs.Length, perPairLosses);
    }
}
