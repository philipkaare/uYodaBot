using YodaTransformer;
using Xunit;

public class ModelSerializerTests
{
    private static TransformerModel NewModel(Random rng)
    {
        var vocab = new Vocabulary();
        var pairs = TrainingData.GetPairs(vocab);
        int maxSeqLen = pairs.Max(p => Math.Max(p.input.Length, p.target.Length));
        return new TransformerModel(vocab.Size, 8, 8, 16, maxSeqLen, rng);
    }

    [Fact]
    public void Stream_RoundTrip_PreservesWeightsAndEpochs()
    {
        var saved = NewModel(new Random(1));

        using var ms = new MemoryStream();
        ModelSerializer.Save(saved, epochs: 4242, ms);
        ms.Position = 0;

        var loaded = NewModel(new Random(2)); // different init weights
        bool ok = ModelSerializer.TryLoad(loaded, out int epochs, ms);

        Assert.True(ok);
        Assert.Equal(4242, epochs);
        Assert.Equal(saved.Wout[0], loaded.Wout[0]);
        Assert.Equal(saved.Embedding.TokenWeights[3], loaded.Embedding.TokenWeights[3]);
        Assert.Equal(saved.Block.Attention.Wq[0], loaded.Block.Attention.Wq[0]);
        Assert.Equal(saved.Block.Ffn.b1, loaded.Block.Ffn.b1);
    }

    [Fact]
    public void Stream_TryLoad_RejectsBadMagic()
    {
        using var ms = new MemoryStream(new byte[] { 1, 2, 3, 4, 0, 0, 0, 0 });
        var model = NewModel(new Random(3));
        bool ok = ModelSerializer.TryLoad(model, out _, ms);
        Assert.False(ok);
    }
}

public class ForwardCacheTests
{
    [Fact]
    public void Forward_PopulatesCaches_UsedByVerboseView()
    {
        var vocab = new Vocabulary();
        var pairs = TrainingData.GetPairs(vocab);
        int maxSeqLen = pairs.Max(p => Math.Max(p.input.Length, p.target.Length));
        var model = new TransformerModel(vocab.Size, 8, 8, 16, maxSeqLen, new Random(42));

        int[] tokens = vocab.Encode("i am hungry");
        float[][] logits = model.Forward(tokens);

        Assert.Equal(tokens.Length, logits.Length);
        Assert.Equal(vocab.Size, logits[0].Length);
        Assert.Equal(tokens.Length, model.Embedding.LastOutput.Length);
        Assert.Equal(tokens.Length, model.Block.Attention.Weights.Length);
        Assert.Equal(tokens.Length, model.Block.FfnOut.Length);
        Assert.Equal(16, model.Block.Ffn.HRelu[0].Length);
    }
}
