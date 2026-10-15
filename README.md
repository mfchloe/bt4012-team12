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

Run in order from the `notebooks/` folder:

1. `01_data_exploration.ipynb`: dataset structure, labels, data-quality issues
2. `02_data_cleaning.ipynb`: removing duplicates and preparing labels
