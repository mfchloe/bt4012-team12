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

To reproduce the final comparison, run `02`, then `03`–`05`, then `06`, `07`, `09` and `10`, following their dependencies above. `01` and `08` provide exploratory analysis and are not required by `11`. Complete the prediction-export sections in `06`, `07` and `09`; in `10`, enable `RUN_TEST = True` in setup before running its final test section, which also exports the test predictions. All four models must provide both validation and test prediction files before running `11`.

Notebook `11` reads saved predictions without retraining. Its setup starts with `RUN_TEST = False`, but its final test cell explicitly sets `RUN_TEST = True`, so **Run All evaluates test at the end**. To skip test evaluation, change the assignment in that final cell to `False`. Re-running an unchanged evaluation reproduces results; test feedback must not guide model or threshold changes.

## Evaluation

Shared code in `src/evaluation.py` and the comparison functions in `11` keep evaluation consistent:

- **Train** on `train_sample` (about 50k pairs, sampled by article). Level-1 scores used by a stacked model are out-of-fold, grouped by article, so no row is scored by a model that saw its article.
- **Choose thresholds** on `validation`: `11` selects each model's best F1 cutoff among the searched candidates. Its additional low-false-alarm comparison selects a separate cutoff per model that flags **at most 1% of genuine validation pairs**, excluding boundary ties when necessary. These cutoffs are applied unchanged to test; actual test false-positive rates may differ from 1%.
- **Report** genuine-versus-manipulated image–caption detection in `11`: ROC-AUC, precision, recall, F1, confusion matrices, ROC curves and detection rates by manipulation type. Precision, recall, F1, confusion matrices and manipulation-type rates use the F1 cutoffs. The separate low-false-alarm table reports actual test false-positive rates and recall at the stricter cutoffs; it does not assume that 1% of posts are fake. Individual modelling notebooks contain additional metrics, robustness checks and prevalence scenarios.
- **Test:** final test results are reported in `11` using models and thresholds fixed before test evaluation. The test set contains 50,705 pairs: 33,829 manipulated (66.7%) and 16,876 genuine (33.3%).
- Dataset shortcuts found in `03` and `04` (absolute ELA, caption length) are listed in `src/feature_sets.ARTIFACT_FEATURES` and excluded from the models.

Each model saves its predictions to `results/predictions/<model>_<split>.parquet` with `row_id` and `p_image_fake` and/or `p_text_fake`, plus `<model>_thresholds.csv`. The prediction prefixes are `text_model`, `image_model`, `early_fusion` and `multimodal_stacking`; splits are `validation` and `test`. `11` reads the prediction files, combines component scores using the same rule for every approach, and selects its own comparison cutoffs from validation rather than loading the threshold CSVs.

**Final test results** (`11`; precision, recall and F1 use validation-selected F1 cutoffs):

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline Model (Text + Image combined) | 0.781 | 0.739 | 0.920 | 0.819 |
| Simple multimodal model (09) | 0.828 | 0.744 | 0.940 | 0.831 |
| Stacking (10) | 0.879 | 0.801 | 0.920 | 0.856 |

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
