# Image-Only Model (`notebooks/07_image_model.ipynb`)

This model predicts `p_image_tampered`, the probability that a news **image itself** was visually manipulated (`face_swap` or `face_attribute`, with or without a caption edit). It uses pixels only. It is an **image-level** score, not a pair-level fake-news prediction: an untouched photo with a swapped or sentiment-flipped caption is a negative.

## The story

1. **Classical image features first.** Traditional forensic cues, each compared between a face and the ring of pixels around it (`04_image_features`):
   - edge strength (Laplacian)
   - local noise (median-filter and Gaussian high-pass residuals)
   - JPEG compression differences (ELA)

   These cues are easy to visualise and explain. An L1 (lasso) logistic regression on train keeps the useful ones. Logistic Regression and XGBoost are then trained on exactly those columns and rows.
2. **Then add CLIP image embeddings** to the same two classifiers. The multimodal models use CLIP, so this keeps the comparison fair: a multimodal model can only beat this baseline by using the caption, not by having CLIP.
3. **Pick the better "+ embeddings" classifier on validation** as the baseline. On a tie, Logistic Regression is kept as the simpler model.

## Results (validation ROC-AUC)

Both classifiers use the same rows, the same split and fixed settings. From the full run on 20 Nov 2026.

| Features | Logistic Regression | XGBoost |
|---|---|---|
| Engineered only (34 kept by the lasso, of 40) | 0.630 | **0.652** |
| Engineered + CLIP image embeddings (802) | **0.815** | 0.771 |

- **Selected baseline:** Logistic Regression on engineered features + CLIP image embeddings.
  - Validation: AUC 0.815, EER 0.272.
  - Thresholds: 0.399 (best F1: precision 0.673, recall 0.886, F1 0.765) and 0.918 (1% false positives; catches 23% of edited images).
- **Pruning:** the lasso (C = 0.1) cost nothing. All 40 features give 0.631 / 0.651.
- **Effect of the new features:** adding the 10 new `image_filters` features to the original 30 raised validation AUC by +0.014 (LR) and +0.009 (XGBoost), in a development check on validation with the same rows and settings.
- **Test:** not evaluated. `RUN_TEST = False`; test scores are saved, but test labels are not used. Set `RUN_TEST = True` once, at the end.

## Inputs

**Engineered candidates (40):**
- 30 columns from the `image` group (`04`). The five absolute-ELA columns are excluded as the known Guardian shortcut (`src/feature_sets.ARTIFACT_FEATURES`).
- 10 columns from the `image_filters` group (`04`, sections 1–2):
  - `img_lap_p90ratio_{min,max,largest}` and `img_noise_p90ratio_{min,max,largest}`: the 90th percentile of the Laplacian or noise response inside the face, divided by the same in the ring. Edits smooth away a face's sharpest details; a mean averages those away, a high percentile keeps them.
  - `img_gauss_ratio_{min,max,largest,spread}`: the Gaussian high-pass residual (σ = 2) inside the face, divided by the ring. It captures texture at a coarser scale than the 3×3 filters.

**Dropped by the lasso:** `img_ela_ratio_mean`, `img_ela_ratio_spread`, `img_lap_ratio_min`, `img_lap_ratio_largest`, `img_lap_ratio_spread`, `img_noise_face_min`.

**Embeddings:** `clip_img_*`, the 768 ViT-L/14 whole-image numbers from `05` (`data/features/clip_emb_*.parquet`). These are the same vectors the multimodal models use.

**Never used:**
- captions and the CLIP caption embedding `clip_txt_*`
- the image–caption similarity groups `clip` and `clip_face`
- file paths, ids, `fake_cls`, `fake_image_box`, `fake_text_pos`

`src/image_model.check_model_inputs` enforces this, and it is unit-tested.

## Method

- **Labels.** `fake_cls` is split on `&`. `image_tampered = 1` if the tags include `face_swap` or `face_attribute`. Anything outside the nine DGM4 combinations raises an error. The label matches `02`'s `image_fake` on every row.
- **Splits.** The team's splits from `02`:
  - Fit once per distinct `train_sample` image (44,857 images).
  - Choose the lasso `C` by 5-fold cross-validation on train, grouped by article.
  - Choose the classifier and thresholds on `validation` (22,126 rows).
  - Test once at the end.
- **Classifiers** (`src/image_model.make_classifiers`, seed 4012, no tuning). The median imputer and scaler are fit on train only, inside each pipeline.
  - Logistic Regression: median imputation → standardisation → `LogisticRegression(max_iter=2000)`.
  - XGBoost: median imputation → 300 trees, depth 4, learning rate 0.05, 0.8 row and column subsampling.
- **Lasso.** Median imputation → standardisation → L1 logistic regression (`liblinear`). `C` is tried over (0.001, 0.01, 0.1, 1). The chosen value is the smallest `C` whose grouped-CV AUC is within 0.005 of the best. One caveat: a linear model picks the features for both classifiers.
- **No usable face.** The `image_filters` columns would be NaN, imputed with the training median. In DGM4 every image has at least 2 faces, so this never happens.

## Reproduce

From the repo root, after `02`, `04` and `05` have been run:

```bash
pip install -r requirements.txt            # includes xgboost
python -m unittest discover tests          # label rule, input check, classifiers, tie rule
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 07_image_model.ipynb
```

- The 07 run takes about 3.5 minutes on a laptop CPU, with no GPU needed.
- Re-running `04` recomputes every image feature, which takes about 10 minutes.

## Outputs

All outputs go to `results/predictions/`:

| File | Contents |
|---|---|
| `image_model_train_sample_oof.parquet` | out-of-fold scores for `train_sample` (article-grouped folds), safe to stack on |
| `image_model_validation.parquet`, `image_model_test.parquet` | scores from the final fit |
| `image_model_thresholds.csv` | `image_fake` (F1) and `image_fake_fpr1` (1% FPR), both from validation |
| `image_model_metrics.csv` | the 2×2 validation table (plus test metrics if `RUN_TEST`) |
| `image_model_config.json` | classifier, kept and dropped features, lasso `C` and CV log, thresholds, versions |

The prediction files have the columns `row_id, split, id, image_tampered, p_image_tampered, p_image_fake`, as before, so `11_model_comparison` and `reports/compare_models.py` read them unchanged.

## References

- Shao, R., Wu, T., & Liu, Z. (2023). Detecting and Grounding Multi-Modal Media Manipulation. *CVPR*. Dataset: https://huggingface.co/datasets/rshaojimmy/DGM4
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). *ICML*. Checkpoint: https://huggingface.co/openai/clip-vit-large-patch14 (embeddings computed in notebook 05)
- Krawetz, N. (2007). A Picture's Worth: Digital Image Analysis and Forensics (error level analysis). *Black Hat*.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *JRSS B*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Libraries: scikit-learn, XGBoost, SciPy, pandas.

## AI-use acknowledgement

Claude Code (Anthropic) was used for the image model. It:
- added the filter functions and the `image_filters` features (`src/image_features.py`)
- simplified `04` to the per-filter original-vs-edited figures, the feature computation and the shortcut checks
- rewrote `notebooks/07_image_model.ipynb`, `src/image_model.py`, `tests/test_image_model.py` and this document
- ran `04`, `07` and the unit tests

All numbers above come from those runs. The team member chose the design (engineered features, then CLIP embeddings, LR vs XGBoost, lasso pruning) and is responsible for checking and explaining the work.
