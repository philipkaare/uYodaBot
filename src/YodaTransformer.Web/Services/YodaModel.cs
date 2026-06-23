using Microsoft.JSInterop;
using YodaTransformer;

namespace YodaTransformer.Web.Services;

public class YodaModel
{
    public const int   DModel       = 8;
    public const int   DHead        = 8;
    public const int   DFF          = 16;
    public const int   Epochs       = 10000;
    public const float LearningRate = 0.005f;

    private const string StorageKey = "yodabot.model.v1";
    private const string ModelUrl   = "yodabot.model";

    private readonly HttpClient _http;
    private readonly IJSRuntime _js;

    public Vocabulary Vocab { get; }
    public (int[] input, int[] target)[] Pairs { get; }
    public int MaxSeqLen { get; }

    public TransformerModel Model { get; private set; }
    public int  TrainedEpochs { get; private set; }
    public bool IsTrained     { get; private set; }

    public YodaModel(HttpClient http, IJSRuntime js)
    {
        _http  = http;
        _js    = js;
        Vocab  = new Vocabulary();
        Pairs  = TrainingData.GetPairs(Vocab);
        MaxSeqLen = Pairs.Max(p => Math.Max(p.input.Length, p.target.Length));
        Model  = NewModel();
    }

    public TransformerModel NewModel() =>
        new(Vocab.Size, DModel, DHead, DFF, MaxSeqLen, new Random(42));

    public async Task InitializeAsync()
    {
        if (await TryLoadFromLocalStorageAsync()) { IsTrained = true; return; }

        try
        {
            byte[] bytes = await _http.GetByteArrayAsync(ModelUrl);
            using var ms = new MemoryStream(bytes);
            var fresh = NewModel();
            if (ModelSerializer.TryLoad(fresh, out int epochs, ms))
            {
                Model = fresh;
                TrainedEpochs = epochs;
                IsTrained = true;
                return;
            }
        }
        catch { /* fall through to untrained */ }

        Model = NewModel();
        TrainedEpochs = 0;
        IsTrained = false;
    }

    public int[] Predict(string sentence)
    {
        int[] tokens = Vocab.Encode(sentence.Trim()); // throws ArgumentException on unknown word
        return Model.Predict(tokens);
    }

    public string Decode(int[] tokens) => Vocab.Decode(tokens);
    public string WordAt(int id)       => Vocab.WordAt(id);

    public void ReplaceModel(TransformerModel trained, int epochs)
    {
        Model = trained;
        TrainedEpochs = epochs;
        IsTrained = true;
    }

    public async Task SaveToLocalStorageAsync()
    {
        using var ms = new MemoryStream();
        ModelSerializer.Save(Model, TrainedEpochs, ms);
        string b64 = Convert.ToBase64String(ms.ToArray());
        await _js.InvokeVoidAsync("localStorage.setItem", StorageKey, b64);
    }

    public async Task<bool> TryLoadFromLocalStorageAsync()
    {
        string? b64 = await _js.InvokeAsync<string?>("localStorage.getItem", StorageKey);
        if (string.IsNullOrEmpty(b64)) return false;
        try
        {
            byte[] bytes = Convert.FromBase64String(b64);
            using var ms = new MemoryStream(bytes);
            var fresh = NewModel();
            if (ModelSerializer.TryLoad(fresh, out int epochs, ms))
            {
                Model = fresh;
                TrainedEpochs = epochs;
                return true;
            }
        }
        catch { /* ignore corrupt storage */ }
        return false;
    }
}
