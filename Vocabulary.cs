namespace YodaTransformer;

public class Vocabulary
{
    public int Size => 3 + _words.Length;
    public const int PAD = 0;
    public const int BOS = 1;
    public const int EOS = 2;

    private Dictionary<string, int> wordToIndex;
    private Dictionary<int, string> indexToWord;
    private readonly string[] _words = { "i", "am", "to", "you", "the", "dark", "side", "hungry", "strong", "are", "will", "join" };

    public Vocabulary()
    {
        wordToIndex = new Dictionary<string, int>();
        indexToWord = new Dictionary<int, string>();

        // Special tokens (indices 0-2)
        wordToIndex["<pad>"] = PAD;
        wordToIndex["<bos>"] = BOS;
        wordToIndex["<eos>"] = EOS;

        indexToWord[PAD] = "<pad>";
        indexToWord[BOS] = "<bos>";
        indexToWord[EOS] = "<eos>";

        // Content words (indices 3-14)
        for (int i = 0; i < _words.Length; i++)
        {
            int idx = 3 + i;
            wordToIndex[_words[i]] = idx;
            // Special case: index 3 ("i") should decode to uppercase "I"
            indexToWord[idx] = (idx == 3) ? "I" : _words[i];
        }
    }

    public int[] Encode(string sentence)
    {
        if (sentence == null)
            throw new ArgumentNullException(nameof(sentence));

        string[] words = sentence.ToLower().Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        int[] result = new int[words.Length + 2];
        result[0] = BOS;

        for (int i = 0; i < words.Length; i++)
        {
            if (!wordToIndex.TryGetValue(words[i], out int index))
            {
                throw new ArgumentException($"Unknown word: {words[i]}");
            }
            result[i + 1] = index;
        }

        result[words.Length + 1] = EOS;
        return result;
    }

    public string Decode(int[] tokens)
    {
        if (tokens == null)
            throw new ArgumentNullException(nameof(tokens));

        System.Collections.Generic.List<string> words = new System.Collections.Generic.List<string>();

        for (int i = 0; i < tokens.Length; i++)
        {
            int token = tokens[i];
            if (token == PAD || token == BOS || token == EOS)
                continue;

            string word = WordAt(token);
            if (word != null)
                words.Add(word);
        }

        return string.Join(" ", words);
    }

    public string WordAt(int index)
    {
        if (indexToWord.TryGetValue(index, out string? word))
            return word;
        return "?";
    }
}
