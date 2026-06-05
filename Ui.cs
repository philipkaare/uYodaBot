namespace YodaTransformer;

public enum MenuOption { Train, Chat, ChatVerbose, Exit }

public class Menu
{
    private readonly bool   _modelLoaded;
    private readonly string _statusLine;

    public Menu(bool modelLoaded, int epochs)
    {
        _modelLoaded = modelLoaded;
        _statusLine  = modelLoaded
            ? $"Model: trained  |  Epochs: {epochs}"
            : "Model: not trained";
    }

    public MenuOption Run()
    {
        var options = new[] { MenuOption.Train, MenuOption.Chat, MenuOption.ChatVerbose, MenuOption.Exit };
        var labels  = new[] { "Train / Retrain", "Chat", "Chat (verbose)", "Exit" };

        int sel = 0;
        // Start on first enabled item
        while (!IsEnabled(options[sel])) sel = (sel + 1) % options.Length;

        while (true)
        {
            Draw(options, labels, sel);
            var key = Console.ReadKey(intercept: true);
            switch (key.Key)
            {
                case ConsoleKey.UpArrow:
                    do sel = (sel - 1 + options.Length) % options.Length;
                    while (!IsEnabled(options[sel]));
                    break;
                case ConsoleKey.DownArrow:
                    do sel = (sel + 1) % options.Length;
                    while (!IsEnabled(options[sel]));
                    break;
                case ConsoleKey.Enter:
                    return options[sel];
                case ConsoleKey.Q:
                    return MenuOption.Exit;
            }
        }
    }

    private bool IsEnabled(MenuOption o) =>
        _modelLoaded || o == MenuOption.Train || o == MenuOption.Exit;

    private void Draw(MenuOption[] options, string[] labels, int sel)
    {
        Console.Clear();
        Console.WriteLine("╔══════════════════════════╗");
        Console.WriteLine("║    u Y o d a B o t       ║");
        Console.WriteLine("╚══════════════════════════╝");
        Console.WriteLine(_statusLine);
        Console.WriteLine();

        for (int i = 0; i < labels.Length; i++)
        {
            bool enabled  = IsEnabled(options[i]);
            bool selected = i == sel;

            if (!enabled)
                Console.ForegroundColor = ConsoleColor.DarkGray;
            else if (selected)
                Console.ForegroundColor = ConsoleColor.Green;

            string prefix = selected ? "> " : "  ";
            string suffix = selected ? " <" : "  ";
            Console.WriteLine($"{prefix}[ {labels[i],-16}]{suffix}");
            Console.ResetColor();
        }

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("↑↓ navigate   Enter select   q quit");
        Console.ResetColor();
    }
}
