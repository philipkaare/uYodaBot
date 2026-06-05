# TUI Menu, Model Persistence, Verbose Training & Inference

**Date:** 2026-06-05
**Status:** Approved

## Goal

Replace the linear startup script with an interactive TUI menu, persist trained weights to disk so training is skipped on subsequent runs, and add verbose training and chat modes that expose the transformer's internals as nicely formatted console output.

---

## New Files

### `TrainingData.cs`
Static class with a single method `GetPairs(Vocabulary vocab)` returning all 23 training pairs. Moves the pair definitions out of `Program.cs`.

### `ModelSerializer.cs`
Static class. Two methods:
- `Save(TransformerModel model, int epochs, string path)` — writes binary file
- `TryLoad(TransformerModel model, out int epochs, string path) → bool` — reads binary file, returns false if file missing or header mismatch

**Binary format:**
- 4-byte magic: `YODA`
- 1-byte version: `1`
- 4-byte int: epoch count
- For each weight matrix (fixed order): `[int rows][int cols][float* rows×cols]` row-major

**Matrix order:** `Embedding.TokenWeights`, `Wout`, then per attention head: `Wq Wk Wv Wo`, then FFN: `W1`, `b1` (serialised as 1×N), `W2`, `b2`.

File written to `yodabot.model` in the working directory.

### `Ui.cs`
`Menu` class. Draws the box-drawn TUI, handles `↑`/`↓` arrow keys and `Enter`. Returns the chosen `MenuOption` enum value. Items for which the model is required (Chat, Chat verbose) are rendered dimmed and skipped if no model is loaded.

**Layout:**
```
╔══════════════════════════╗
║      u Y o d a B o t     ║
╚══════════════════════════╝
Model: trained  |  Epochs: 10000

  [ Train / Retrain ]
> [ Chat            ] <
  [ Chat (verbose)  ]
  [ Exit            ]

↑↓ navigate   Enter select   q quit
```

Status line shows `Model: not trained` and dims Chat options when no save file exists.

### `VerboseTrainer.cs`
Wraps `Trainer`. Registers an `OnEpoch` callback and uses `Console.SetCursorPosition` to rewrite a fixed-height panel every N epochs (configurable, default 200).

**Panel layout:**
```
Training — 10000 epochs
══════════════════════════════════════════
Epoch 1000 / 10000  [██████████░░░░░░░░░░]  10%

Loss curve (last 6 checkpoints):
  2.69 ████████████████████
  1.31 ████████████░░░░░░░░
  0.38 ████░░░░░░░░░░░░░░░░  ←

Per-pair losses:
  "i am hungry"            ██░░░░░░  0.12
  "you will join..."       ████░░░░  0.41 ▲
  ...

Press Q to skip remaining epochs
```

- Bar widths normalised to initial loss (epoch 0).
- `▲` flags pairs above the current average loss.
- `Q` keypress (non-blocking `Console.KeyAvailable` poll) skips remaining epochs and jumps to save.
- On completion: panel freezes, appends `✓ Done — loss 0.0004`.

### `VerboseInference.cs`
Static `Run(TransformerModel model, Vocabulary vocab, string sentence)`. Calls `model.Forward` then reads stored intermediate values via new public accessors. Prints the following numbered sections:

**① Tokenise**
> Each word is mapped to an integer ID from the vocabulary. Special tokens `<bos>` (start) and `<eos>` (end) are added automatically.

Shows token words and IDs in aligned columns.

**② Embed + Positional Encoding (16-dim)**
> Each token ID is looked up in a learned embedding table (16 floats). A fixed sinusoidal positional encoding is added so the model knows the order of tokens.

Shows all positions, first 8 values of each 16-dim vector with `...` suffix.

**③ Attention — Head N / 4 (4-dim queries/keys)**
> Each head learns to attend to different token relationships. The weight matrix shows how much each position (row) attends to each other position (col) after softmax normalisation.

Shows the full softmax weight matrix (seqLen × seqLen) for each of the 4 heads, with token labels on rows and columns.

**④ Output logits → predictions**
> The transformer's output vectors are projected to vocabulary size via Wout. The highest logit at each position becomes the output token.

For each non-special output position, shows a ranked bar chart of top-3 tokens:
```
pos 1 — top tokens:
  hungry  ████████████████████  4.21  ✓
  i       ████░░░░░░░░░░░░░░░░  0.34
  am      ██░░░░░░░░░░░░░░░░░░  0.11
```
Bar widths normalised to the max logit in that position. `✓` marks the selected token.

---

## Modified Files

### `AttentionHead.cs`
Add public read-only properties exposing stored forward-pass values:
- `public float[][] Weights => _weights;` (softmax attention weights)
- `public float[][] Q => _q;`, `public float[][] K => _k;`, `public float[][] V => _v;`

### `EmbeddingLayer.cs`
Store the last forward output: add `private float[][] _lastOutput;` and `public float[][] LastOutput => _lastOutput;`. Assign in `Forward`.

### `Trainer.cs`
Add optional callback: `public Action<int, float, float[]>? OnEpoch { get; set; }`. Invoked at the end of each epoch with `(epochIndex, averageLoss, perPairLosses[])`. `VerboseTrainer` sets this; the default `null` means no overhead in non-verbose mode.

### `Program.cs`
Rewritten as a thin dispatcher:
1. Construct `vocab`, `model`, `trainer`
2. `ModelSerializer.TryLoad(model, out int epochs, "yodabot.model")`
3. Loop: show `Menu`, dispatch on result:
   - `Train` → `VerboseTrainer.Run(trainer, pairs, epochs)` → `ModelSerializer.Save(...)`
   - `Chat` → existing chat loop
   - `ChatVerbose` → chat loop using `VerboseInference.Run`
   - `Exit` → break

---

## Startup Behaviour

| Condition | Menu state |
|-----------|-----------|
| `yodabot.model` exists, valid | `Model: trained \| Epochs: N`, all items enabled |
| File missing | `Model: not trained`, Chat items dimmed and skipped by arrow nav |
| File corrupt/wrong version | Print warning, treat as missing |

---

## Files Summary

| File | Change |
|------|--------|
| `TrainingData.cs` | New |
| `ModelSerializer.cs` | New |
| `Ui.cs` | New |
| `VerboseTrainer.cs` | New |
| `VerboseInference.cs` | New |
| `AttentionHead.cs` | Add public accessors |
| `EmbeddingLayer.cs` | Store + expose LastOutput |
| `Trainer.cs` | Add OnEpoch callback |
| `Program.cs` | Rewrite as dispatcher |
