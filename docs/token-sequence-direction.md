# Token Sequence Direction

The project direction is to keep raw examples easy to tokenize, then convert
them into a shared token-sequence training form.

```text
raw text / image / audio / label-like data
  -> modality-specific tokenizer or encoder
  -> TokenSequence
  -> shared prediction objective
```

The common layer is not Signal JSONL. The common layer is the token sequence
that training consumes.

Input and output adapters may stay modality- or task-specific. The part to
share is the Transformer core that consumes hidden sequences.

```text
input adapter
  -> shared hidden sequence
  -> shared Transformer core
  -> output adapter / head
```

For Fashion-MNIST this means the classification-head baseline can coexist with
the token-continuation direction:

```text
image path + label id
  -> ImagePatchAdapter
  -> SharedTransformerCore
  -> ClassificationHead

image path + label text
  -> image/text token or embedding adapters
  -> SharedTransformerCore
  -> text-token continuation scoring
```

## Core Shape

```text
TokenSequence:
  token_ids
  optional loss_mask
```

`loss_mask` is optional. A plain language-modeling sequence can train on all
next-token positions. A prompted classification task can mask loss to the answer
tokens only.

## Selection Classification

Selection classification should be represented as candidate continuation
scoring, not as a special classification head by default.

```text
image tokens + prompt tokens + candidate label tokens
```

The evaluator compares candidate-label continuation losses.

```text
loss("Sneaker" | image tokens, "Class:")
loss("Bag" | image tokens, "Class:")
```

This keeps classification close to the same sequence-prediction form used for
text.

## Raw Examples

Raw persisted data should stay close to the source and task.

```text
text:
  text

image classification:
  image path or reference
  label text or label id

audio:
  audio path or reference
  optional transcript text
```

Do not add generic `channel`, `role`, `kind`, or `target` fields just to make
unrelated tasks look uniform. Add structure only when a tokenizer, encoder,
training objective, or evaluator actually uses it.

## Legacy Direction

Signal JSONL and signal-tag rendering are transitional experiment surfaces.
They should not receive new feature work unless needed to retire an existing
experiment safely.
