# BT4012 Team 12: Multimodal News Manipulation Detection

Detecting manipulated news image-caption pairs in the [DGM4](https://huggingface.co/datasets/rshaojimmy/DGM4) dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download and unpack the dataset into `data/raw/DGM4` (about 10.7GB download, 11GB on disk):

```bash
python scripts/download_data.py --clear-cache
```

- `--clear-cache` deletes the zips from the Hugging Face cache after unpacking, so the data is not stored twice.
- `--only infoswap bbc` fetches only the listed image archives (metadata is always fetched). Useful for a quick start.
- Re-running is safe: archives already unpacked are skipped.

`data/` is git-ignored. After the download:

```
data/raw/DGM4/
├── metadata/        train.json, val.json, test.json
├── origin/          bbc, guardian, usa_today, washington_post
└── manipulation/    infoswap, simswap, HFGI, StyleCLIP
```

The `image` field in the metadata is relative to `data/raw/`, e.g. `data/raw/DGM4/origin/bbc/0272/120.jpg`.

## Notebooks

Use the project `.venv` kernel and run cells from `notebooks/`. Dependencies are saved files, not notebook numbering.

| Notebook | Purpose | Inputs |
|---|---|---|
| **Data** | | |
| [01_data_exploration](notebooks/01_data_exploration.ipynb) | Understand labels, duplicates, captions and faces | Raw metadata; image cache for previews |
| [02_data_cleaning](notebooks/02_data_cleaning.ipynb) | Clean tables, targets and article-level training sample | Raw metadata |
| **Features** | | |
| [03_text_features](notebooks/03_text_features.ipynb) | Caption structure, named entities, sentiment and GPT-2 fluency (`txt_*`) | `02` tables |
| [04_image_features](notebooks/04_image_features.ipynb) | Image and face forensics: ELA, Laplacian and noise residuals (`img_*`) | `02` tables and images |
| [05_clip_features](notebooks/05_clip_features.ipynb) | CLIP embeddings and image–text consistency features | `02` tables and images |
| **Single-mode models** | | |
| [06_text_model](notebooks/06_text_model.ipynb) | Text-only detector (`text_fake`) | `02`, `03`, `05` caption embeddings |
| [07_image_model](notebooks/07_image_model.ipynb) | Image-only detector (`image_fake`): frozen CLIP + logistic regression; lightweight ViT-B/32 version about 6× faster. See [docs/image_baseline.md](docs/image_baseline.md) | `02` tables and images; `05` embeddings (`data/interim/clip/`, `data/features/clip_emb_*`) |
| **Multimodal models** | | |
| [08_multimodal_feature_analysis](notebooks/08_multimodal_feature_analysis.ipynb) | Phrase–image and name–face feature experiments; writes `mm_phrase` and `mm_name_face` | `02`, `05` outputs and face caches |
| [09_multimodal_early_fusion](notebooks/09_multimodal_early_fusion.ipynb) | Linear and neural early fusion on CLIP embeddings | `02`, `05` embeddings; optional saved `10` predictions. Saves the final model's predictions for `11` |
| [10_multimodal_stacking](notebooks/10_multimodal_stacking.ipynb) | Late-fusion stacking: CLIP classifiers + gradient-boosting judge | `02`, `03`, `04`, `05` outputs |
| **Comparison** | | |
| [11_model_comparison](notebooks/11_model_comparison.ipynb) | All models on the same rows, metrics and thresholds; single-mode late fusion vs multimodal | Saved predictions in `results/predictions/` |

`05` does not depend on `03` or `04`, so the three feature notebooks can run in any order.
