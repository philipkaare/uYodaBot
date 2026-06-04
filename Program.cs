// Yoda Transformer — top-level entry point
const int vocabSize  = 13;
const int dModel     = 8;
const int dHead      = 8;
const int dFF        = 16;
const int maxSeqLen  = 10;
const int epochs     = 3000;
const float learningRate = 0.005f;

var rng     = new Random(42);
var vocab   = new YodaTransformer.Vocabulary();
var model   = new YodaTransformer.TransformerModel(vocabSize, dModel, dHead, dFF, maxSeqLen, rng);
var trainer = new YodaTransformer.Trainer(model, learningRate);

// Build training pairs — each (input, target) is encoded with BOS/EOS.
var pairs = new (int[] input, int[] target)[]
{
    (vocab.Encode("i went to school"),          vocab.Encode("to school i went")),
    (vocab.Encode("i am hungry"),               vocab.Encode("hungry i am")),
    (vocab.Encode("you went to the dark side"), vocab.Encode("to the dark side you went")),
    (vocab.Encode("i am you"),                  vocab.Encode("you i am")),
    (vocab.Encode("you went to school"),        vocab.Encode("to school you went")),
    (vocab.Encode("i went to the dark side"),   vocab.Encode("to the dark side i went")),
};

// Training loop
float loss = 0f;
for (int epoch = 0; epoch < epochs; epoch++)
{
    loss = trainer.TrainStep(pairs);
    if (epoch % 200 == 0)
        Console.WriteLine($"Epoch {epoch,4}: loss = {loss:F4}");
}
Console.WriteLine($"Epoch {epochs,4}: loss = {loss:F4}");

// Inference
string[] tests = { "i went to school", "i am hungry", "you went to the dark side" };
Console.WriteLine("\n--- Yoda Transformer Inference ---");
foreach (var sentence in tests)
{
    int[] tokens    = vocab.Encode(sentence);
    int[] predicted = model.Predict(tokens);
    Console.WriteLine($"  Input:  {sentence}");
    Console.WriteLine($"  Output: {vocab.Decode(predicted)}");
    Console.WriteLine();
}

// Attention note
Console.WriteLine("(Attention weights are stored in model.Block.Attention — inspect in debugger)");
