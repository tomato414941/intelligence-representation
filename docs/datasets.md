# Datasets

This document records the intended role of datasets in this project. It is not
a downloader guide. Generated corpora, extracted samples, run metrics, and
checkpoints remain local artifacts unless explicitly versioned.

## Supported And Candidate Datasets

| Dataset | Modality | Approximate size | Status | Description |
| --- | --- | ---: | --- | --- |
| Tiny Shakespeare | text | about 1 MB | supported | A single small Shakespeare text corpus commonly used for toy language-model examples. |
| WikiText-2 | text | about 2M tokens | candidate | A small Wikipedia-derived language-modeling corpus with train, validation, and test splits. |
| WikiText-103 | text | about 103M tokens | candidate | A larger Wikipedia-derived language-modeling corpus built from full articles. |
| TinyStories | text | over 2M stories | candidate | A synthetic corpus of short English stories written with simple vocabulary and grammar. |
| OpenWebText | text | about 8M documents / 40 GB text | candidate | An open reproduction of GPT-2-style WebText, collected from web pages linked by Reddit posts. |
| FineWeb-Edu | text | about 1.3T tokens | candidate | A filtered educational subset of FineWeb built from Common Crawl web pages. |
| FineWeb-Edu score-2 | text | about 5.4T tokens | candidate | A broader FineWeb-Edu variant using a lower educational-score threshold. |
| MNIST | image | 70k images | supported | A grayscale handwritten digit image dataset with 10 classes. |
| Fashion-MNIST | image | 70k images | supported | A grayscale fashion-product image dataset with 10 classes. |
| CIFAR-10 | image | 60k images | supported | A color natural-image dataset with 10 classes. |

## Current Policy

| Policy area | Decision |
| --- | --- |
| Local artifacts | Downloaded datasets and generated samples stay under local artifact paths such as `data/` or `runs/`. |
| Large text data | Prefer streaming or fixed-size slices before introducing full-dataset workflows. |
| Evidence level | Loss reduction on toy data is a smoke signal. Stronger claims require task-specific validation on non-toy data. |
| Scope control | Add dataset-specific code only when an actual training or evaluation path needs it. |
