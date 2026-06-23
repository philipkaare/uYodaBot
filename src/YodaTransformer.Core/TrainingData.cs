namespace YodaTransformer;

public static class TrainingData
{
    public static (int[] input, int[] target)[] GetPairs(Vocabulary vocab) => new[]
    {
        (vocab.Encode("i am hungry"),                   vocab.Encode("hungry i am")),
        (vocab.Encode("i am you"),                      vocab.Encode("you i am")),
        (vocab.Encode("you are hungry"),                vocab.Encode("hungry you are")),
        (vocab.Encode("you will join the dark side"),   vocab.Encode("join the dark side you will")),
        (vocab.Encode("i am strong"),                   vocab.Encode("strong i am")),
        (vocab.Encode("you are strong"),                vocab.Encode("strong you are")),
        (vocab.Encode("i will join the dark side"),     vocab.Encode("join the dark side i will")),
        (vocab.Encode("you will join"),                 vocab.Encode("join you will")),
        (vocab.Encode("i will join"),                   vocab.Encode("join i will")),
        (vocab.Encode("i am the dark side"),            vocab.Encode("the dark side i am")),
        (vocab.Encode("you are the dark side"),         vocab.Encode("the dark side you are")),
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
        (vocab.Encode("the light side are strong"),      vocab.Encode("strong the light side are")),
        (vocab.Encode("the light side are hungry"),      vocab.Encode("hungry the light side are")),
        (vocab.Encode("the light side will join you"),   vocab.Encode("join you the light side will")),
        (vocab.Encode("the light side will join i"),     vocab.Encode("join i the light side will")),
        (vocab.Encode("the light side of the force are strong"),      vocab.Encode("strong the light side of the force are")),
        (vocab.Encode("the light side of the force are hungry"),      vocab.Encode("hungry the light side of the force are")),
        (vocab.Encode("the light side of the force will join you"),   vocab.Encode("join you the light side of the force will")),
        (vocab.Encode("the light side of the force will join i"),     vocab.Encode("join i the light side of the force will")),
    };
}
