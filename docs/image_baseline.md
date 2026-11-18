# Image-Only Model (`notebooks/08_image_model.ipynb`)

This model predicts `p_image_tampered`, the probability that a news **image itself** was visually manipulated (`face_swap` or `face_attribute`, with or without a caption edit). It uses pixels only. It is an **image-level** score, not a pair-level fake-news prediction: an untouched photo with a swapped or sentiment-flipped caption is a negative.

Two versions are built, using the same method with different frozen CLIP encoders:

| | Main model | Lightweight model |
|---|---|---|
| Whole images | ViT-L/14 | ViT-B/32 |
| Face crops | ViT-B/16 | ViT-B/32 |
| Embeddings | computed in notebook `04`, reused | computed in 08 |
| Test ROC-AUC / F1 | **0.854 / 0.789** | 0.820 / 0.770 |
| Speed on an Apple-silicon Mac | about 4 images/s | about 25–27 images/s (about 6× faster) |
| Use | best accuracy | when speed matters |

## Reproduce

From the repo root (tested on macOS, Apple M-series, MPS):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt nbconvert

# DGM4 metadata + the 8 image zips into the Hugging Face cache (~10.7 GB).
# src/images.py reads JPGs straight from these zips, so nothing needs unpacking.
# Do NOT use download_data.py --clear-cache: it deletes the zips the notebooks read.
python -c "import sys; sys.path.insert(0, 'scripts'); from download_data import *; [download(f) for f in METADATA_FILES + ARCHIVES]"

# Shared splits and row_id (writes data/processed/*.parquet)
cd notebooks && jupyter nbconvert --to notebook --execute 02_data_cleaning.ipynb --output-dir ../data/interim/runs && cd ..

python -m unittest discover tests                      # label, crop and pooling rules (14 tests)

cd notebooks
IMAGE_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute 08_image_model.ipynb --output-dir ../data/interim/runs/smoke   # a few minutes
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 08_image_model.ipynb                 # full run
```

The last command is equivalent to *Restart & Run All* in Jupyter.

**Inputs from notebook `04`.** The main model reads `04`'s embeddings:
- `data/interim/clip/images.npy` and `data/interim/clip/faces_vitb16.npy`
- `data/features/clip_emb_*.parquet`, used for an alignment check

Run `04`, or copy those files from a teammate. If they are missing, 08 computes the same embeddings itself with the shared `src/clip_features` code. That takes several hours on a laptop, is cached in `data/interim/image_model/`, and resumes if interrupted.

**Run times on this Mac:**
- **First run:** about 70 minutes, to embed everything with ViT-B/32 for the lightweight model. Embeddings are cached in chunks, so an interrupted run resumes where it stopped.
- **Later runs:** about 5 minutes, measured with every embedding already cached. This includes about 2 minutes for the speed benchmark, about 1.5 minutes for model fitting, and under 30 seconds for the bootstrap confidence intervals.
- **One-off downloads:** the ViT-L/14 (~1.7 GB) and ViT-B/16 (~600 MB) weights, used for the benchmark and the check.

Notes:
- On macOS the notebook needs MPS for a reasonable runtime. If `torch.backends.mps.is_available()` is `False`, it falls back to CPU, which is much slower.
- The smoke run samples 40 rows of every `fake_cls` per split (seed 4012) and writes only to `data/interim/image_model/smoke/`.

## Setup recorded by the run

| Item | Value |
|---|---|
| Dataset | [`rshaojimmy/DGM4`](https://huggingface.co/datasets/rshaojimmy/DGM4), revision `9efd3f1ca14cb1dac1692cdce0beea6844b968a6` |
| Main model checkpoints | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) (whole images) and [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16) (faces), frozen. Embeddings from notebook `04` |
| Lightweight checkpoint | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32), revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, frozen |
| Seed | `RANDOM_STATE = 4012` (same as `02`, `04`, `05`) |
| Python | 3.11.18 |
| Packages | torch 2.15.0, transformers 5.23.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.4.6, pillow 13.0.0, huggingface_hub 2.4.0, datasets 5.2.0, pyarrow 26.0.1 |

Each run writes the checkpoint revisions and versions to `results/predictions/image_model_config.json` and `image_model_light_config.json`.

## Method

- **Labels.** `fake_cls` is split on `&`. `image_tampered = 1` if the tags include `face_swap` or `face_attribute`, else 0. Anything outside the nine DGM4 combinations raises an error (`src/image_model.py`). The label matches `02`'s `image_fake` on every row.
- **Splits.** The team's shared splits from `02`:
  - Fit on `train_sample` (49,943 rows, 44,857 distinct images, sampled by article).
  - Choose `C` by 5-fold CV grouped by article `id`.
  - Choose the variant and thresholds on `validation` (22,126 rows).
  - Evaluate each model once on `test` (50,705 rows).
  - The classifier is fit once per distinct image. Predictions are reported for every original row.
- **Inputs.** L2-normalised CLIP image embeddings only. Captions, paths, ids, `fake_image_box` and `fake_text_pos` are never used. The path alone gives the label away.
- **Face crops.** The same rule is applied to every image, real or fake, and to both models. `fake_image_box` is never read by the model.
  - Uses the provided `mtcnn_boxes`, clipped to the image.
  - Keeps boxes that are at least 16 px on both sides (284,963 crops).
  - Adds a 20% margin to each box (`src/clip_features.crop_face`).
  - Averages the crop embeddings per image, then re-normalises.
  - Falls back to the whole image, embedded with the face encoder, if no crop is valid (1 image of 110,030).
  - Why crop and average (face EDA, section 4.1, `train_sample`):
    - Images have 2.59 valid faces on average, the same for both classes, so face count isn't a shortcut.
    - A typical face is about 33 px wide after CLIP's 224 px resize: about 2×2 ViT-L/14 patches, or one ViT-B/32 patch.
    - The edited face is the largest in only 66.5% of manipulated images, so all faces are averaged rather than keeping only the largest.
- **`04`'s embeddings were checked before use.**
  - The ViT-L/14 vectors are identical to `04`'s per-row copies in `data/features/clip_emb_*.parquet`.
  - 200 random face crops were re-embedded with ViT-B/16 and match `04`'s vectors (cosine 1.0000).
- **Variants.** For each model, a `StandardScaler` + `LogisticRegression` is fit on one of:
  - `full_image`: the whole-image embedding
  - `face_mean`: the averaged face-crop embedding
  - `full_plus_face`: both, concatenated
- **Not done.** XGBoost (optional in the proposal) was not run.

## Results

From the full run of `08_image_model.ipynb` on 18 Nov 2026. The lightweight model's predictions are byte-identical to the earlier ViT-B/32-only run, and the main model reproduces the earlier comparison numbers.

**Validation AUC, used for variant selection**

| Variant | Main | Lightweight |
|---|---|---|
| `full_image` | 0.801 | 0.745 |
| `face_mean` | 0.812 | 0.795 |
| **`full_plus_face`** (selected for both) | **0.848** (C = 1) | **0.813** (C = 0.1) |

**Test, each model at its own thresholds chosen on validation**

| Model | Threshold | ROC-AUC | EER | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Majority class (always "manipulated") | – | 0.500 | 0.500 | 0.531 | 0.531 | 1.000 | 0.693 |
| **Main**, F1 threshold | 0.411 | **0.854** | **0.239** | **0.749** | **0.712** | **0.884** | **0.789** |
| Main, 1% FPR threshold | 0.936 | 0.854 | 0.239 | 0.614 | 0.976 | 0.279 | 0.434 |
| Lightweight, F1 threshold | 0.435 | 0.820 | 0.269 | 0.722 | 0.688 | 0.874 | 0.770 |
| Lightweight, 1% FPR threshold | 0.924 | 0.820 | 0.269 | 0.574 | 0.961 | 0.205 | 0.338 |

Confusion matrices at the F1 threshold:

| | Main: predicted untouched | Main: predicted manipulated | Light: predicted untouched | Light: predicted manipulated |
|---|---|---|---|---|
| untouched (23,801) | 14,196 | 9,605 | 13,120 | 10,681 |
| manipulated (26,904) | 3,122 | 23,782 | 3,395 | 23,509 |

**Breakdowns on test** (main model, F1 threshold unless stated)
- **Misses:** recall is 0.867 for `face_swap` and 0.902 for `face_attribute`. By generator: SimSwap 0.777, HFGI 0.877, InfoSwap 0.914, StyleCLIP 0.929.
- **False positives on text-only edits:**

  | Class | Main, F1 threshold | Main, 1% FPR | Light, F1 threshold |
  |---|---|---|---|
  | `orig` | 0.406 | 0.009 | 0.448 |
  | `text_swap` | 0.408 | 0.005 | 0.472 |
  | `text_attribute` | 0.369 | 0.006 | 0.398 |

  Text-only edits are flagged no more often than originals, as expected for a pixel-only model.
- **Face count:** `full_plus_face` AUC is 0.865 with two valid faces and 0.824 with four or more, the dilution expected from averaging.
- **Publisher** (main vs light): Guardian 0.899 vs 0.877, BBC 0.866 vs 0.819, USA Today 0.743 vs 0.706, Washington Post 0.725 vs 0.679. The main model's gain is smallest on the Guardian, so it does not come from the Guardian re-compression shortcut.

**Uncertainty** (bootstrap over test articles, 1,000 samples; `image_model_confidence_intervals.csv`): main AUC 0.854 (95% CI 0.850–0.857); lightweight AUC 0.820 (0.816–0.823); the main model's gain of 0.034 (0.031–0.037) holds in every sample.

**Speed** (section 11.2): 500 validation images scored from scratch on this Mac's GPU, from reading the JPG to the final score. The ranges cover three runs.

| | Main | Lightweight |
|---|---|---|
| Image-encoder parameters | 390 M | 88 M |
| Total time per image | about 225–250 ms | about 38–40 ms |
| Images per second | about 4–4.5 | about 25–27 |
| One image at a time (latency) | about 240–290 ms | about 45 ms |
| 100,000 images | about 6 hours | about 1 hour |

- **Where the time goes:** almost all of it is CLIP itself. Reading and cropping take about 8 ms per image for both models, and the logistic regression under 0.1 ms.
- **Why the absolute times are not comparable across machines:** they depend on the hardware. The roughly 6× ratio between the two models is what carries over.

**Data checks**

| Check | Result |
|---|---|
| Rows (metadata) | train 208,184, val 22,126, test 50,705. `02` removes 51,015 repeated `orig` rows from train, leaving 157,169 |
| Distinct images | train 141,189, val 19,909, test 45,264 |
| Image label balance | about 53–54% of rows, about 60% of distinct images |
| Missing / unreadable JPGs | 0 / 0 across all 206,362 distinct images |
| Shared article ids or image paths across splits | 0 |
| Byte-identical images (MD5) across splits | train–val 24, train–test 56, val–test 7 |
| Test rows sharing an exact image with `train_sample` | 18; excluding them changes the metrics by at most 0.0001 |
| Identical files carrying conflicting labels | 0 |

The split was not changed. `04` also reports near-duplicate photos found by CLIP similarity; those were not re-checked here.

## Outputs

All outputs are in `results/predictions/`:

| File | Contents |
|---|---|
| `image_model_train_sample_oof.parquet` | **main model**, out-of-fold scores for `train_sample`, safe to stack on |
| `image_model_validation.parquet`, `image_model_test.parquet` | main model, scores from the final fit |
| `image_model_thresholds.csv` | main model: `image_fake` (F1) and `image_fake_fpr1` (1% FPR), both from validation |
| `image_model_metrics.csv` | main model: validation selection table and test metrics |
| `image_model_config.json` | main model: variant, `C`, thresholds, CV log, revisions and versions |
| `image_model_light_*.parquet`, `image_model_light_thresholds.csv`, `image_model_light_config.json` | the same for the lightweight model |
| `image_model_comparison.csv` | section 11: all six variants on validation, both models on test, and the speed table |
| `image_model_confidence_intervals.csv` | 95% bootstrap confidence intervals for both models' test AUC and F1, and for the gap between them |

The prediction parquets have the columns `row_id, split, id, image_tampered, p_image_tampered, p_image_fake`. `row_id` is the team key from `02`. `p_image_fake` duplicates `p_image_tampered` under the name `05` and `src/evaluation.evaluate` expect, so the fusion model can join on `row_id` directly.

**These are image-level scores repeated on every row that shows the image, not pair-level fake-news predictions.**

The caches in `data/interim/image_model/` (~0.8 GB) are gitignored. They hold the file check, the ViT-B/32 embeddings and their index files.

## Report-ready summary

> **Image-only model.** We embed each news image with frozen CLIP encoders and train a logistic regression to predict whether the image was manipulated (a face swap or face attribute edit), ignoring the caption. The main model uses the larger encoders computed in notebook 04: ViT-L/14 for the whole image and ViT-B/16 for face crops. Face crops come from the dataset's MTCNN detections, applied with the same rule to real and manipulated images. Their embeddings are averaged per image and concatenated with the whole-image embedding. Combining both views works best (validation AUC 0.848, against 0.801 for the whole image and 0.812 for faces alone). On the held-out test set, the main model reaches ROC-AUC 0.854 and EER 0.239. At the validation-chosen F1 threshold it has accuracy 0.749, precision 0.712, recall 0.884 and F1 0.789; the majority-class baseline has accuracy 0.531 and F1 0.693. Face-attribute edits are caught more often than face swaps (recall 0.902 vs 0.867), and SimSwap is the hardest generator. Untouched images with edited captions are flagged no more often than untouched originals, which confirms the model reacts to pixels rather than text.
>
> **Lightweight model.** When detection speed matters (screening every photo as it is posted, or running on modest hardware), the same method with the small ViT-B/32 encoder is about 6× faster on our machine (about 26 vs 4 images per second). It reaches test AUC 0.820 and F1 0.770. It loses most on the whole image (validation AUC 0.745 vs 0.801), because its 32-pixel patches see a typical 33-pixel face as a single patch; the face crops recover most of that. A practical setup is two-stage: the lightweight model screens everything, and the main model re-checks only what it flags.
>
> **Limitations.** Mean-pooling dilutes the signal when only one of several faces is edited: AUC drops from 0.865 with two faces to 0.824 with four or more. Performance varies by publisher, from 0.73 to 0.90 AUC. The highest AUC is for the Guardian, whose fakes were re-compressed when DGM4 was built, so part of the signal may be a dataset artefact. At a realistic 1% false-positive rate the main model catches only 28% of manipulated images. The encoders are frozen, the model was trained on a third of the training set, and it has not been tested on generators outside DGM4. Speed figures come from one machine.

## References

- Shao, R., Wu, T., & Liu, Z. (2023). Detecting and Grounding Multi-Modal Media Manipulation. *CVPR*. Dataset: https://huggingface.co/datasets/rshaojimmy/DGM4, code and documentation: https://github.com/rshaojimmy/MultiModal-DeepFake
- Shao, R., Wu, T., Wu, J., Nie, L., & Liu, Z. (2024). Detecting and Grounding Multi-Modal Media Manipulation and Beyond. *IEEE TPAMI*.
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). *ICML*. Checkpoints: https://huggingface.co/openai/clip-vit-large-patch14 and https://huggingface.co/openai/clip-vit-base-patch16 (main model; embeddings computed in notebook 04), https://huggingface.co/openai/clip-vit-base-patch32 (lightweight model)
- Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks (MTCNN). *IEEE Signal Processing Letters*. The face boxes are provided with DGM4; we did not run a detector.
- Ojha, U., Li, Y., & Lee, Y. J. (2023). Towards Universal Fake Image Detectors that Generalize Across Generative Models. *CVPR*. This is the motivation for linear probes on frozen CLIP features.
- Libraries: Hugging Face `transformers` and `huggingface_hub`, scikit-learn, PyTorch, pandas.
- Team code reused unchanged: `src/images.py`, `src/clip_features.py` and `src/evaluation.py`, plus the splits from `notebooks/02_data_cleaning.ipynb`.
- Team outputs reused: the ViT-L/14 and ViT-B/16 embeddings from `notebooks/04_clip_features.ipynb`, run by Chloe, are the main model's inputs.

## AI-use acknowledgement

Claude Code (Anthropic) was used for the image model. It:
- audited the repository
- wrote `src/image_model.py`, `tests/test_image_model.py`, `notebooks/08_image_model.ipynb` and this document
- added a pointer to this document in `README.md`
- ran the download, `02`, the unit tests, the smoke run and the full runs on a team member's machine
- checked that `04`'s embeddings line up with 08's images and faces, and moved the shared `04` files into `data/features/` and `data/interim/clip/`
- restructured the notebook so the main model uses `04`'s embeddings, with ViT-B/32 as a lightweight alternative, and added the accuracy and speed comparison.

All numbers above come from those runs. The team member reviewed the design choices (fitting on `train_sample`, the exported column names, using `04`'s encoders for the main model) and is responsible for checking and explaining the work.
