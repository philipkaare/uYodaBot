// Yoda Transformer — top-level entry point

const int dModel     = 8;
const int dHead      = 8;
const int dFF        = 16;
const int maxSeqLen  = 10;
const int epochs     = 10000;
const float learningRate = 0.005f;

var rng     = new Random(42);
var vocab   = new YodaTransformer.Vocabulary();
var model   = new YodaTransformer.TransformerModel(vocab.Size, dModel, dHead, dFF, maxSeqLen, rng);
var trainer = new YodaTransformer.Trainer(model, learningRate);

// Build training pairs — each (input, target) is encoded with BOS/EOS.
var pairs = new (int[] input, int[] target)[]
{
    (vocab.Encode("i am hungry"),               vocab.Encode("hungry i am")),
    (vocab.Encode("i am you"),                  vocab.Encode("you i am")),
    (vocab.Encode("you are hungry"),            vocab.Encode("hungry you are")),
    (vocab.Encode("you will join the dark side"), vocab.Encode("join the dark side you will")),
    (vocab.Encode("i am strong"),               vocab.Encode("strong i am")),
    (vocab.Encode("you are strong"),            vocab.Encode("strong you are")),
    (vocab.Encode("i will join the dark side"), vocab.Encode("join the dark side i will")),
    (vocab.Encode("you will join"),             vocab.Encode("join you will")),
    (vocab.Encode("i will join"),               vocab.Encode("join i will")),
    (vocab.Encode("i am the dark side"),        vocab.Encode("the dark side i am")),
    (vocab.Encode("you are the dark side"),     vocab.Encode("the dark side you are")),

    // additional pairs within existing vocabulary
    (vocab.Encode("you are i"),                     vocab.Encode("i you are")),
    (vocab.Encode("i am to the dark side"),         vocab.Encode("to the dark side i am")),
    (vocab.Encode("you are to the dark side"),      vocab.Encode("to the dark side you are")),
    (vocab.Encode("i am the side"),                 vocab.Encode("the side i am")),
    (vocab.Encode("you are the side"),              vocab.Encode("the side you are")),
    (vocab.Encode("i will join you"),               vocab.Encode("join you i will")),
    (vocab.Encode("i will join the side"),          vocab.Encode("join the side i will")),
    (vocab.Encode("you will join the side"),        vocab.Encode("join the side you will")),
    (vocab.Encode("the dark side are strong"),      vocab.Encode("strong the dark side are")),
    (vocab.Encode("the dark side are hungry"),      vocab.Encode("hungry the dark side are")),
    (vocab.Encode("the dark side will join you"),   vocab.Encode("join you the dark side will")),
    (vocab.Encode("the dark side will join i"),     vocab.Encode("join i the dark side will")),
};

// Training loop
float loss = 0f;
for (int epoch = 0; epoch < epochs; epoch++)
{
    (loss, _) = trainer.TrainStep(pairs);
    if (epoch % 200 == 0)
        Console.WriteLine($"Epoch {epoch,4}: loss = {loss:F4}");
}
Console.WriteLine($"Epoch {epochs,4}: loss = {loss:F4}");

// Inference
string[] tests = { "i am hungry", "you are strong", "you will join the dark side" };
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
