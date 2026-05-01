# Datasets

This document records the intended role of datasets in this project. It is not
a downloader guide. Generated corpora, extracted samples, run metrics, and
checkpoints remain local artifacts unless explicitly versioned.

## Supported And Candidate Datasets

| Dataset | Modality | Approximate size | Status | Intended use |
| --- | --- | ---: | --- | --- |
| Tiny Shakespeare | text | about 1 MB | supported | Fast text language-model checks only. It should not be treated as strong evidence of text modeling quality. |
| WikiText-2 | text | about 2M tokens | candidate | Lightweight language-model evaluation when a stable public benchmark is useful. |
| WikiText-103 | text | about 103M tokens | candidate | Larger language-model evaluation without moving to web-scale data. |
| TinyStories | text | over 2M stories | candidate | Text generation quality checks for small models, especially coherent short-form English. |
| OpenWebText | text | about 8M documents / 40 GB text | candidate | GPT-2-style web-text pretraining checks when a large but finite local corpus is desired. |
| FineWeb-Edu | text | about 1.3T tokens | candidate | Main large text source for sampled or streamed pretraining experiments. Use slices, not the full dataset, for this project scale. |
| FineWeb-Edu score-2 | text | about 5.4T tokens | candidate | Future broader web-text source if FineWeb-Edu slices are too narrow. Not a near-term default. |
| MNIST | image | 70k images | supported | Simple grayscale image classification and image-to-label experiments. |
| Fashion-MNIST | image | 70k images | supported | Grayscale image classification, image-text choice, and image-text answer experiments. |
| CIFAR-10 | image | 60k images | supported | Small color image classification checks. |

## Current Policy

| Policy area | Decision |
| --- | --- |
| Local artifacts | Downloaded datasets and generated samples stay under local artifact paths such as `data/` or `runs/`. |
| Large text data | Prefer streaming or fixed-size slices before introducing full-dataset workflows. |
| Evidence level | Loss reduction on toy data is a smoke signal. Stronger claims require task-specific validation on non-toy data. |
| Scope control | Add dataset-specific code only when an actual training or evaluation path needs it. |
