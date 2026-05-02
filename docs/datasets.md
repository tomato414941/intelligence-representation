# Datasets

This document records datasets that are already supported or under discussion
for this project. It is not a downloader guide.

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
| Mini-ImageNet | image | about 65k images | candidate | A smaller ImageNet-derived image classification dataset commonly organized around 100 classes. |
| Food-101 | image | about 101k images | candidate | A food image classification dataset with 101 classes. |
| Places365 | image | large | candidate | A scene recognition dataset with 365 place categories. |
| ImageNet-1K | image | about 1.28M train images | candidate | A large object image classification dataset with 1000 classes. |
| iNaturalist 2021 | image | large | candidate | A fine-grained species image dataset with many biological categories. |
| Qhapaq computer shogi KIF records | game / shogi | 18,948 games | supported | Computer-shogi game records from Qhapaq Research Lab, downloaded as a KIF archive and converted to local game-record JSONL for shogi move-choice experiments. |

## Notes

| Area | Note |
| --- | --- |
| Local artifacts | Downloaded datasets, generated samples, run metrics, and checkpoints are usually local artifacts under paths such as `data/` or `runs/`. |
| Large text data | Large datasets can be used through streaming or fixed-size slices before deciding whether full local copies are needed. |
| Evidence level | Tiny or toy datasets are useful for quick checks, but larger or task-specific datasets are needed for stronger evaluation claims. |
| Qhapaq raw data | The local raw archive is `data/qhapaq/raw/180913_kif_rota.7z`; source pages include `https://www.qhapaq.org/shogi/kifdb/` and `https://www.qhapaq.org/shogi/`. |
| Qhapaq processed data | The local source-derived records are `data/qhapaq/processed/qhapaq_all_games.jsonl`; train/eval files should be split at game boundaries before generating move-choice examples. |
