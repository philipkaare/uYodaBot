# Multi-Head Attention + Extended Training Data

**Date:** 2026-06-05  
**Status:** Approved

## Goal

Improve Yoda-speak output quality by replacing the single attention head with 4-head attention (dHead=4 each) and enabling 12 additional training pairs that were commented out.

## Architecture

### New: `Layers/MultiHeadAttention.cs`

Wraps 4 `AttentionHead(dModel=16, dHead=4)` instances.

- **Forward(x):** Runs each head independently, sums their outputs. Mathematically equivalent to concatenate-then-project with a block-diagonal output weight matrix.
- **Backward(dOut):** Backprops through each head (each receives the full `dOut`), accumulates per-head weight gradients internally, returns summed `dX`.
- **ApplyGradients(lr, maxNorm):** Applies clipped SGD to all 16 weight matrices (4 heads × Wq/Wk/Wv/Wo) using the same `ClipAndUpdate` logic currently in `Trainer`.

`AttentionHead` is unchanged — it remains the single-head primitive.

### Modified: `Layers/TransformerBlock.cs`

- Field type: `AttentionHead` → `MultiHeadAttention`
- Constructor: `dHead` parameter → `numHeads`
- `Backward` return type: drop the 4 attention weight gradient matrices. New signature: `(float[][] dX, float[][] dW1, float[] db1, float[][] dW2, float[] db2)`
- Public property `Attention` changes type to `MultiHeadAttention`

### Modified: `TransformerModel.cs`

- Constructor: `dHead` → `numHeads`
- `Backward` return type: drop attention grads to match block. New signature: `(float[][] dWout, float[][] dW1, float[] db1, float[][] dW2, float[] db2, float[][] dTokenWeights)`

### Modified: `Trainer.cs`

- Backward destructure: remove `dWq, dWk, dWv, dWo`
- After `_model.Backward(...)`: call `_model.Block.Attention.ApplyGradients(_lr, 1.0f)`
- Remove the 4 `ClipAndUpdate` calls for attention weights

### Modified: `Program.cs`

- `const int dHead = 16` → `const int nHeads = 4`
- Constructor call updated: `dHead` → `nHeads`
- Uncomment 12 additional training pairs (all use existing vocabulary tokens)

## Gradient Flow Summary

```
Trainer.TrainStep
  → _model.Backward(tokens, dLogits)
      → _block.Backward(dBlockOut)           returns (dX, dW1, db1, dW2, db2)
          → _attention.Backward(dAttnOut)    returns dX only, stores dWq/dWk/dWv/dWo per head
          → _ffn.Backward(dFfnOut)           returns (dXNorm2, dW1, db1, dW2, db2)
      → _embedding.Backward(tokens, dEmbOut)
  → _model.Block.Attention.ApplyGradients(lr, 1.0f)   ← new step
  → ClipAndUpdate Wout, W1, b1, W2, b2, embedding
```

## Files Changed

| File | Change |
|------|--------|
| `Layers/MultiHeadAttention.cs` | **New** |
| `Layers/TransformerBlock.cs` | Swap head type, drop attn grads from return |
| `TransformerModel.cs` | Drop attn grads from return, rename param |
| `Trainer.cs` | Add ApplyGradients call, remove 4 manual attn updates |
| `Program.cs` | Rename dHead→nHeads, uncomment training pairs |
