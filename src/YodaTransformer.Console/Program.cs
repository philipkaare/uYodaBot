// uYodaBot — top-level entry point

const string ModelPath    = "yodabot.model";
const int    DModel       = 8;
const int    DHead        = 8;
const int    DFF          = 16;
const int    Epochs       = 10000;
const float  LearningRate = 0.005f;

var rng   = new Random(42);
var vocab = new YodaTransformer.Vocabulary();
var pairs = YodaTransformer.TrainingData.GetPairs(vocab);

// Size the positional encoding to the longest sequence in the training data
// (inputs and targets), so it stays correct as the data grows.
int MaxSeqLen = pairs.Max(p => Math.Max(p.input.Length, p.target.Length));

YodaTransformer.TransformerModel model = NewModel();

bool modelLoaded = YodaTransformer.ModelSerializer.TryLoad(model, out int savedEpochs, ModelPath);
if (!modelLoaded)
{
    savedEpochs = 0;
    // A model on disk that fails to load is incompatible (e.g. the vocabulary
    // changed size). Delete it so we start over with a fresh model.
    if (File.Exists(ModelPath))
    {
        File.Delete(ModelPath);
        Console.Error.WriteLine($"[startup] Removed incompatible model file '{ModelPath}'. Train a new model from the menu.");
    }
}

while (true)
{
    var choice = new YodaTransformer.Menu(modelLoaded, savedEpochs).Run();

    switch (choice)
    {
        case YodaTransformer.MenuOption.Train:
            // Always retrain from scratch with a fresh model.
            rng     = new Random(42);
            model   = NewModel();
            var trainer = new YodaTransformer.Trainer(model, LearningRate);

            YodaTransformer.VerboseTrainer.Run(trainer, pairs, vocab, Epochs);
            YodaTransformer.ModelSerializer.Save(model, Epochs, ModelPath);
            modelLoaded  = true;
            savedEpochs  = Epochs;
            Console.WriteLine("Model saved. Press any key to return to menu...");
            Console.ReadKey(intercept: true);
            break;

        case YodaTransformer.MenuOption.Chat:
            RunChat(model, vocab, verbose: false);
            break;

        case YodaTransformer.MenuOption.ChatVerbose:
            RunChat(model, vocab, verbose: true);
            break;

        case YodaTransformer.MenuOption.Exit:
            return;
    }
}

YodaTransformer.TransformerModel NewModel() =>
    new(vocab.Size, DModel, DHead, DFF, MaxSeqLen, rng);

static void RunChat(
    YodaTransformer.TransformerModel model,
    YodaTransformer.Vocabulary       vocab,
    bool verbose)
{
    Console.Clear();
    Console.WriteLine(verbose
        ? "--- Yoda Chat (verbose) — type a sentence, or 'quit' to exit ---"
        : "--- Yoda Chat — type a sentence, or 'quit' to exit ---");
    Console.WriteLine($"  Known words: {string.Join(", ", vocab.Words)}");
    Console.WriteLine();

    while (true)
    {
        Console.Write("You: ");
        string? input = Console.ReadLine();
        if (input is null || input.Trim().ToLower() is "quit" or "exit" or "q")
            break;
        if (string.IsNullOrWhiteSpace(input)) continue;

        if (verbose)
        {
            YodaTransformer.VerboseInference.Run(model, vocab, input.Trim());
        }
        else
        {
            try
            {
                int[] tokens    = vocab.Encode(input.Trim());
                int[] predicted = model.Predict(tokens);
                Console.WriteLine($"Yoda: {vocab.Decode(predicted)}");
                Console.WriteLine();
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine($"  (unknown word — {ex.Message})");
                Console.WriteLine();
            }
        }
    }
}
