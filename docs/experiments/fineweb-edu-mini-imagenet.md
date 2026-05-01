# FineWeb-Edu And Mini-ImageNet Experiment

## Purpose

This experiment moves the project beyond toy text and 10-class image datasets
while keeping the scope small enough to run and inspect.

## Dataset Scope

| Modality | Dataset | Scope |
| --- | --- | --- |
| text | FineWeb-Edu | A streamed or fixed-size text slice, not the full dataset. |
| image | Mini-ImageNet | Image classification data with broader class diversity than MNIST, Fashion-MNIST, or CIFAR-10. |

## Initial Runs

| Run | Input | Training path | Check |
| --- | --- | --- | --- |
| text LM | FineWeb-Edu slice | text corpus -> tokenizer -> causal text model | Train/eval loss, checkpoint restore, generated text sample. |
| image classification | Mini-ImageNet | image -> shared image route -> classification head | Train/eval loss and accuracy. |

## Success Criteria

| Area | Criterion |
| --- | --- |
| data | The selected text slice and image split can be prepared without committing generated data. |
| text | The text run completes on GPU and produces a checkpoint that can be restored for generation. |
| image | The image run completes on GPU and improves evaluation accuracy over the initial model. |
| scope | The experiment reuses existing training paths unless a missing dataset adapter is clearly required. |

## Non-Goals

| Area | Non-goal |
| --- | --- |
| full scale | Training on the full FineWeb-Edu or full ImageNet-scale data is not part of the first run. |
| architecture | This experiment does not introduce a new multimodal architecture. |
| capability claim | Passing this experiment does not prove a general predictive representation system. |
