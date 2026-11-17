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
| `01_data_exploration` | Understand labels, duplicates, captions and faces | Raw metadata; image cache for previews |
| `02_data_cleaning` | Clean tables, targets and article-level training sample | Raw metadata |
| `03_feature_engineering` | Caption structure/sentiment/fluency and image/face forensics | `02` tables; images for Part 2 |
| `04_clip_features` | CLIP embeddings and image–text consistency features | `02` tables and images; independent of `03` |
| `05_multimodal_stacking` | Train the existing multimodal detector | `02`, `03`, `04` outputs |
| `06_text_model` | Train the specialised text-only detector | `02`, `03` text, relevant `04` outputs |
| [07_multimodal_analysis](notebooks/07_multimodal_analysis.ipynb) | Phrase–image consistency features and validation comparisons | Matching `02` and `04` outputs |

For developing `07`: **02 → 04 → 07**; reuse matching team outputs instead of rerunning extraction. `07` follows the saved-feature convention of `03`/`04`, writes `mm_phrase` tables, and includes integration notes for `05`. Its optional tiny function-check cell can be deleted without affecting the normal workflow.
