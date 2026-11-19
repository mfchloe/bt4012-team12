"""Frozen-embedding early-fusion experiments. No downloads or training at import."""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import best_f1_threshold, binary_metrics, combine_fake_score

TARGETS = ("image_fake", "text_fake")
DEFAULT_TRAINING = dict(batch_size=256, learning_rate=.001, weight_decay=.0001,
                        patience=5, min_delta=.0001)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    temporary.replace(path)


def align_embeddings(labels, embeddings, expected_columns=None):
    """Exact pair coverage, label row order and matching embedding dimension order."""
    for name, frame in [("labels", labels), ("embeddings", embeddings)]:
        if not frame.columns.is_unique:
            raise ValueError(f"Duplicate {name} columns")
        if frame.row_id.isna().any() or not frame.row_id.is_unique:
            raise ValueError(f"Missing or duplicate {name} row IDs")
    if set(labels.row_id) != set(embeddings.row_id):
        raise ValueError("Embedding and metadata row coverage differs")
    img = [c for c in embeddings if c.startswith("clip_img_")]
    txt = [c for c in embeddings if c.startswith("clip_txt_")]
    if not img or len(img) != len(txt) or set(embeddings) != {"row_id", *img, *txt}:
        raise ValueError("Expected only row_id and equally sized image/text embeddings")
    # Both modalities must list corresponding dimensions in the same order.
    if [c.removeprefix("clip_img_") for c in img] != [c.removeprefix("clip_txt_") for c in txt]:
        raise ValueError("Image/text embedding dimension order differs")
    if expected_columns is not None and (img, txt) != expected_columns:
        raise ValueError("Embedding column order differs between splits")
    ordered = embeddings.set_index("row_id").loc[labels.row_id]
    image, text = ordered[img].to_numpy(dtype=np.float32), ordered[txt].to_numpy(dtype=np.float32)
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("Nonfinite embeddings")
    for target in TARGETS:
        if set(labels[target].unique()) != {0, 1}:
            raise ValueError(f"Both binary classes required for {target}")
    if not np.array_equal(labels.is_fake.to_numpy(), labels[list(TARGETS)].max(axis=1).to_numpy()):
        raise ValueError("is_fake is inconsistent with modality targets")
    return image, text, (img, txt)


def joint_features(image, text, mode="concat"):
    image, text = np.asarray(image, dtype=np.float32), np.asarray(text, dtype=np.float32)
    if image.ndim != 2 or image.shape != text.shape or not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("Expected equally shaped finite image/text matrices")
    blocks = [image, text]
    if mode == "comparisons":
        blocks += [image * text, np.abs(image - text)]
    elif mode != "concat":
        raise ValueError(f"Unknown feature mode {mode}")
    return np.concatenate(blocks, axis=1).astype(np.float32)


def feature_names(columns, mode):
    img, txt = columns
    return img + txt + (["product_" + c.removeprefix("clip_img_") for c in img]
                        + ["absdiff_" + c.removeprefix("clip_img_") for c in img] if mode == "comparisons" else [])


def grouped_holdout(labels, seed=4012):
    if labels.id.isna().any() or labels.id.nunique() < 5:
        raise ValueError("At least five nonmissing article groups are required")
    fit, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
                        .split(labels, groups=labels.id))
    if set(labels.iloc[fit].id) & set(labels.iloc[holdout].id):
        raise ValueError("Article leakage in internal split")
    for indices in (fit, holdout):
        if any(labels.iloc[indices][target].nunique() != 2 for target in TARGETS):
            raise ValueError("An internal split lacks a target class; expand data")
    return fit, holdout


class FusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(.2),
                                    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 2))

    def forward(self, inputs):
        return self.layers(inputs)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def best_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def predict_network(model, features, device="cpu", batch_size=256):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            parts.append(model(batch).sigmoid().cpu().numpy())
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite network predictions")
    return result


def train_network(features, targets, *, epochs=30, seed=4012, device="cpu", holdout=None,
                  batch_size=256, learning_rate=.001, weight_decay=.0001, patience=5,
                  min_delta=.0001, verbose=True):
    """Early stop on internal BCE only; restore best weights. No holdout => fixed epochs."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    x, y = np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)
    if y.shape != (len(x), 2) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected finite features and two binary targets")
    # pandas/Arrow can expose read-only arrays; PyTorch expects writable backing storage.
    if not x.flags.writeable:
        x = x.copy()
    if not y.flags.writeable:
        y = y.copy()
    seed_everything(seed)
    model = FusionMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    best_loss, best_epoch, best_state, stale = np.inf, 0, None, 0
    records, started = [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(batch)
        row = {"epoch": epoch, "train_loss": total / len(x), "elapsed_seconds": time.perf_counter() - started}
        if holdout is not None:
            hx, hy = holdout
            model.eval()
            subtotal = 0.
            with torch.inference_mode():
                for start in range(0, len(hx), batch_size):
                    bx = torch.as_tensor(hx[start:start + batch_size], dtype=torch.float32, device=device)
                    by = torch.as_tensor(hy[start:start + batch_size], dtype=torch.float32, device=device)
                    subtotal += loss_fn(model(bx), by).item() * len(bx)
            current = subtotal / len(hx)
            if not np.isfinite(current):
                raise ValueError("Nonfinite internal holdout loss")
            row["holdout_loss"] = current
            if current < best_loss - min_delta:
                best_loss, best_epoch = current, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        records.append(row)
        if verbose:
            print(f"epoch {epoch:02d}/{epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)
        if holdout is not None and stale >= patience:
            break
    if holdout is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model, pd.DataFrame(records), best_epoch


def fit_linear(features, targets, fit, holdout, candidates=(.001, .01, .1, 1.)):
    """Choose each C on article-held-out training rows, then refit scaler/heads on full train."""
    inner_scaler = StandardScaler().fit(features[fit])
    xf = inner_scaler.transform(features[fit]).astype(np.float32)
    xh = inner_scaler.transform(features[holdout]).astype(np.float32)
    chosen, selection = [], []
    for j, target in enumerate(TARGETS):
        aucs = []
        for c in candidates:
            model = LogisticRegression(C=c, max_iter=3000).fit(xf, targets[fit, j])
            auc = roc_auc_score(targets[holdout, j], model.predict_proba(xh)[:, 1])
            selection.append({"target": target, "C": c, "internal_AUC": auc})
            aucs.append(auc)
        chosen.append(candidates[int(np.argmax(aucs))])
    scaler = StandardScaler().fit(features)
    x = scaler.transform(features).astype(np.float32)
    heads = [LogisticRegression(C=c, max_iter=3000).fit(x, targets[:, j]) for j, c in enumerate(chosen)]
    return scaler, heads, pd.DataFrame(selection)


def predict_linear(scaler, heads, features):
    x = scaler.transform(features).astype(np.float32)
    return np.column_stack([head.predict_proba(x)[:, 1] for head in heads])


def score_targets(probabilities):
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected two probabilities per pair")
    return {"image_fake": p[:, 0], "text_fake": p[:, 1], "is_fake": combine_fake_score(p[:, 0], p[:, 1])}


def genuine_fpr_threshold(score, genuine, rate=.01):
    score, genuine = np.asarray(score), np.asarray(genuine, dtype=bool)
    if score.shape != genuine.shape or not genuine.any() or not np.isfinite(score).all() or not 0 <= rate < 1:
        raise ValueError("Expected aligned finite scores, genuine rows and rate in [0,1)")
    values = np.sort(score[genuine])
    return float(np.nextafter(values[len(values) - int(np.floor(rate * len(values))) - 1], np.inf))


def validation_tables(labels, predictions):
    rows = []
    genuine = labels.is_fake.to_numpy() == 0
    for name, probabilities in predictions.items():
        for target, score in score_targets(probabilities).items():
            if len(score) != len(labels):
                raise ValueError("Prediction/label lengths differ")
            for point, threshold in [("F1", best_f1_threshold(labels[target], score)),
                                      ("FPR1", genuine_fpr_threshold(score, genuine))]:
                rows.append({"experiment": name, "target": target, "operating_point": point,
                             **binary_metrics(labels[target], score, threshold),
                             "genuine_FPR": float((score[genuine] >= threshold).mean())})
    return pd.DataFrame(rows)


def saved_reference(path, labels):
    """Bad or unavailable historical reference never prevents controlled 09 comparisons."""
    path = Path(path)
    if not path.exists():
        return None, "Saved 10 reference unavailable: prediction file missing"
    try:
        frame = pd.read_parquet(path)
        if not frame.row_id.is_unique or set(frame.row_id) != set(labels.row_id):
            raise ValueError("prediction row IDs do not exactly cover current validation")
        p = frame.set_index("row_id").loc[labels.row_id, ["p_image_fake", "p_text_fake"]].to_numpy()
        score_targets(p)
        return p, "Saved 10 reference aligned. Historical feature/model provenance is not re-established."
    except (ValueError, KeyError, OSError) as exc:
        return None, f"Saved 10 reference omitted: {exc}"


def paired_bootstrap(labels, predictions, metrics, comparisons, repetitions=500, seed=4012):
    """Conditional paired article intervals; fixed fitted models and validation thresholds."""
    prepared, draws, points = {}, {}, {}
    lookup = metrics.set_index(["experiment", "target", "operating_point"])
    codes, articles = pd.factorize(labels.id, sort=False)
    if len(articles) < 2 or repetitions < 2:
        raise ValueError("Need at least two articles and resamples")
    used = set(name for pair in comparisons for name in pair)
    for name in used:
        for target, s in score_targets(predictions[name]).items():
            y = labels[target].to_numpy().astype(bool)
            order = np.argsort(s, kind="stable")
            starts = np.r_[0, np.flatnonzero(np.diff(s[order])) + 1]
            pred = s >= lookup.loc[(name, target, "F1"), "threshold"]
            strict = s >= lookup.loc[(name, target, "FPR1"), "threshold"]
            key = name, target
            prepared[key] = order, starts, y, pred, strict
            points[key] = np.array([lookup.loc[(name, target, "F1"), "AUC"],
                                    lookup.loc[(name, target, "F1"), "F1"],
                                    lookup.loc[(name, target, "FPR1"), "Recall"]])
            draws[key] = np.full((repetitions, 3), np.nan)
    rng = np.random.default_rng(seed)
    for b in range(repetitions):
        weights = np.bincount(rng.integers(0, len(articles), len(articles)), minlength=len(articles))[codes]
        for key, (order, starts, y, pred, strict) in prepared.items():
            positive = np.add.reduceat(weights[order] * y[order], starts)
            negative = np.add.reduceat(weights[order] * ~y[order], starts)
            denominator = positive.sum() * negative.sum()
            auc = np.sum(positive * (np.cumsum(negative) - .5 * negative)) / denominator if denominator else np.nan
            tp, fp, fn = weights[y & pred].sum(), weights[~y & pred].sum(), weights[y & ~pred].sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
            recall = weights[y & strict].sum() / weights[y].sum() if weights[y].sum() else np.nan
            draws[key][b] = auc, f1, recall
        if (b + 1) % 100 == 0:
            print(f"Paired article bootstrap {b+1}/{repetitions}", flush=True)
    records = []
    for name, reference in comparisons:
        for target in (*TARGETS, "is_fake"):
            delta = draws[(name, target)] - draws[(reference, target)]
            point = points[(name, target)] - points[(reference, target)]
            for j, metric in enumerate(("AUC", "F1", "Recall_at_FPR1")):
                valid = delta[:, j][np.isfinite(delta[:, j])]
                if len(valid) < .95 * repetitions:
                    raise ValueError("Too few resamples contain both classes")
                low, high = np.quantile(valid, [.025, .975])
                records.append(dict(experiment=name, reference=reference, target=target, metric=metric,
                                    difference=point[j], CI_low=low, CI_high=high, valid_resamples=len(valid)))
    return pd.DataFrame(records)
