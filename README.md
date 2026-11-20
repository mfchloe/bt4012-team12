# BT4012 Team 12: Multimodal News Manipulation Detection

Detecting manipulated news image-caption pairs in the [DGM4](https://huggingface.co/datasets/rshaojimmy/DGM4) dataset. A pair is fake if its image was edited (face swap or face attribute) or its caption was edited (caption swapped from another article, or sentiment words flipped).

**Question:** does a model that looks at the image and caption *together* detect manipulation better than two separate single-mode models? The project compares a text-only model, an image-only model, the **Baseline Model (Text + Image combined)**, which combines them, and two multimodal models (a simple multimodal model (early fusion: one model trained on the image and caption embeddings joined together), and stacking) on the same rows and metrics. The baseline uses two separate models: the image-only model looks only at the image and the text-only model only at the caption, and a pair counts as fake if either says its part is fake. Neither sees the other's input, so it cannot check whether a caption matches its photo.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Use the `.venv` kernel for the notebooks. On a Mac, PyTorch uses the Apple GPU (MPS) when available; CPU works but CLIP embedding is much slower.

## Data

Download the DGM4 metadata and image archives (about 10.7 GB) and unpack them into `data/raw/DGM4`:

```bash
python scripts/download_data.py
```

- `--only infoswap bbc` fetches only the listed image archives (metadata is always fetched).
- Re-running is safe: archives already unpacked are skipped.
- **Do not use `--clear-cache`.** The notebooks read images straight from the zips in the Hugging Face cache (`src/images.py`), and that option deletes them.

`data/` is git-ignored. Notebooks write their outputs there:

| Folder | Contents | Written by |
|---|---|---|
| `data/raw/DGM4/` | `metadata/` (train, val, test JSON), `origin/` and `manipulation/` images | `scripts/download_data.py` |
| `data/processed/` | Cleaned `train`, `train_sample`, `validation`, `test` tables, one row per pair, keyed by `row_id` | `02` |
| `data/features/` | Feature groups as `<group>_<split>.parquet`, keyed by `row_id`: `text`, `image`, `image_filters`, `clip`, `clip_face`, `clip_emb`, `mm_phrase`, `mm_name_face` | `03`, `04`, `05`, `08` |
| `data/interim/` | Caches: CLIP image, face and caption embeddings (`clip/`), word table, model run artifacts | `05`, `06`, `09` |

The `image` field in the metadata is relative to `data/raw/`, e.g. `data/raw/DGM4/origin/bbc/0272/120.jpg`.

## Notebooks

Run cells from `notebooks/`. A notebook needs only the notebooks in its Inputs column to have been run, since everything is passed through saved files.

| Notebook | Purpose | Inputs |
|---|---|---|
| **Data** | | |
| [01_data_exploration](notebooks/01_data_exploration.ipynb) | Labels, duplicates, captions and faces | Raw metadata |
| [02_data_cleaning](notebooks/02_data_cleaning.ipynb) | Clean tables, targets (`is_fake`, `image_fake`, `text_fake`) and the article-level training sample | Raw metadata |
| **Features** | | |
| [03_text_features](notebooks/03_text_features.ipynb) | Caption structure, named entities, sentiment and GPT-2 fluency (`txt_*`) | `02` |
| [04_image_features](notebooks/04_image_features.ipynb) | Image and face forensics: ELA, Laplacian, noise and Gaussian high-pass residuals (`img_*`), with one example output per filter | `02`, images |
| [05_clip_features](notebooks/05_clip_features.ipynb) | CLIP embeddings of images, faces and captions; image–caption similarity; near-duplicate photos across splits | `02`, images |
| **Single-mode models** | | |
| [06_text_model](notebooks/06_text_model.ipynb) | Text-only detector (`text_fake`): TF-IDF, word-level classifier and CLIP caption embedding, stacked with gradient boosting | `02`, `03`, `05` caption embeddings |
| [07_image_model](notebooks/07_image_model.ipynb) | Image-only detector (`image_fake`): logistic regression vs XGBoost on engineered image features (L1-pruned), then the same two with CLIP image embeddings added; the better "+ embeddings" model on validation is the baseline. See [docs/image_baseline.md](docs/image_baseline.md) | `02`, `04`, `05` image embeddings |
| **Multimodal models** | | |
| [08_multimodal_feature_analysis](notebooks/08_multimodal_feature_analysis.ipynb) | Experiments with phrase–image and name–face matching features | `02`, `05` |
| [09_multimodal_early_fusion](notebooks/09_multimodal_early_fusion.ipynb) | Simple multimodal model (early fusion): logistic regression and MLP on joined image and caption embeddings | `02`, `05` |
| [10_multimodal_stacking](notebooks/10_multimodal_stacking.ipynb) | Stacking: classifiers on face, image, caption and image+caption embeddings, combined by gradient boosting | `02`, `03`, `04`, `05` |
| **Comparison** | | |
| [11_model_comparison](notebooks/11_model_comparison.ipynb) | Every model on the same rows, metrics and thresholds | Saved predictions from `06`, `07`, `09`, `10` |

`05` does not depend on `03` or `04`, so the three feature notebooks can run in any order.

## Evaluation

Shared code in `src/evaluation.py` keeps results comparable across notebooks:

- **Train** on `train_sample` (about 50k pairs, sampled by article). Level-1 scores used by a stacked model are out-of-fold, grouped by article, so no row is scored by a model that saw its article.
- **Tune** on `validation`: thresholds are chosen there, both the best-F1 threshold and the one that flags 1% of genuine pairs.
- **Report** AUC, PR-AUC, EER, precision, recall and F1 for each target; the detection rate per manipulation type; results without near-duplicate photos; and alert volumes at realistic fake rates (0.1%, 1%, 5%).
- **Test** once at the end, with models and thresholds fixed on validation (`RUN_TEST = True` in each notebook). The test set has not been evaluated yet.
- Dataset shortcuts found in `03` and `04` (absolute ELA, caption length) are listed in `src/feature_sets.ARTIFACT_FEATURES` and excluded from the models.

Each model saves its predictions to `results/predictions/<model>_<split>.parquet` with `row_id` and `p_image_fake` and/or `p_text_fake`, plus `<model>_thresholds.csv`. `11` reads these files.

**Validation results so far** (`11`):

| Model | `is_fake` AUC | Fake pairs caught at 1% false alarms |
|---|---|---|
| Baseline Model (Text + Image combined) | 0.777 | 19% |
| Simple multimodal model (09) | 0.817 | 34% |
| Stacking (10) | 0.872 | 41% |

## Code

| Path | Contents |
|---|---|
| `src/evaluation.py` | Metrics, thresholds, image + caption score combination (a pair is genuine only if both parts look genuine), detection by type, prevalence tables |
| `src/feature_sets.py` | Feature-group column lists and the excluded shortcut features |
| `src/images.py` | Reads DGM4 images from the zipped archives |
| `src/image_features.py` | Filter functions, forensic maps and per-face image features (`04`) |
| `src/text_word_features.py` | Per-word caption features for the text model (`06`) |
| `src/clip_features.py` | CLIP loading, batched embedding and face crops (`05`) |
| `src/image_model.py` | Labels, input check, classifiers and selection rule for the image model (`07`) |
| `src/early_fusion.py` | Embedding alignment, MLP training and validation tables (`09`) |
| `scripts/download_data.py` | Downloads and unpacks DGM4 |
| `tests/` | Unit tests for `src/image_model.py`: `python -m unittest discover tests` |
