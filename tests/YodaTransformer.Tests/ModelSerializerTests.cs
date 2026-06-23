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
