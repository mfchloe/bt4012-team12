"""CLIP embeddings for DGM4 images, face crops and captions.

Images are loaded and preprocessed in DataLoader worker processes (so the GPU is not left waiting on
JPEG decoding), which is why the datasets live in a module rather than a notebook.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPImageProcessorPil, CLIPModel, CLIPTokenizer

from src.images import load_image

# Whole images and captions: the large model, which separates swapped captions best (04, section 0)
CLIP_NAME = "openai/clip-vit-large-patch14"
# Face crops: ~300k crops make the large model take ~2h; ViT-B/16 takes ~30 min and keeps most of the
# face-level detection signal (04, section 3)
FACE_CLIP_NAME = "openai/clip-vit-base-patch16"
FACE_MARGIN = 0.2  # enlarge each MTCNN box by this fraction on every side, so the crop includes hair and jaw


def best_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(device, model_name=CLIP_NAME):
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    return model, CLIPImageProcessorPil.from_pretrained(model_name), CLIPTokenizer.from_pretrained(model_name)


def crop_face(image, box, margin=FACE_MARGIN):
    width, height = image.size
    x1, y1, x2, y2 = (int(v) for v in box)
    pad_x, pad_y = int(margin * (x2 - x1)), int(margin * (y2 - y1))
    return image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)))


class ImageDataset(Dataset):
    """Whole images (box=None) or face crops, preprocessed for the given CLIP model."""

    def __init__(self, image_paths, boxes=None, model_name=CLIP_NAME):
        self.image_paths = list(image_paths)
        self.boxes = None if boxes is None else list(boxes)
        self.processor = CLIPImageProcessorPil.from_pretrained(model_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image = load_image(self.image_paths[i])
        if self.boxes is not None:
            image = crop_face(image, self.boxes[i])
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def encode_images(model, dataset, device, batch_size=64, num_workers=8):
    """L2-normalised CLIP image embeddings, as a float32 array (n, dim)."""
    # timeout: a stuck worker raises an error instead of hanging the run forever
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        timeout=300 if num_workers > 0 else 0)
    out = []
    for pixels in loader:
        emb = model.get_image_features(pixel_values=pixels.to(device)).pooler_output
        out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu())
    return torch.cat(out).numpy()


def encode_images_chunked(model, dataset, device, cache_prefix, chunk_size=20_000, num_workers=8):
    """encode_images in chunks, saving each chunk as <cache_prefix>_part<k>.npy and logging progress to
    <cache_prefix>_progress.log. Re-running skips chunks already saved, so an interrupted run resumes."""
    cache_prefix = Path(cache_prefix)
    log_path = cache_prefix.with_name(cache_prefix.name + "_progress.log")
    n = len(dataset)
    parts = []
    start = time.time()
    for k, lo in enumerate(range(0, n, chunk_size)):
        part_path = cache_prefix.with_name(f"{cache_prefix.name}_part{k:03d}.npy")
        hi = min(n, lo + chunk_size)
        if not part_path.exists():
            np.save(part_path, encode_images(model, Subset(dataset, range(lo, hi)), device, num_workers=num_workers))
        parts.append(part_path)
        elapsed = (time.time() - start) / 60
        with open(log_path, "a") as log:
            log.write(f"{time.strftime('%H:%M:%S')}  {hi:,}/{n:,} done  ({elapsed:.1f} min this run)\n")
    return np.concatenate([np.load(p) for p in parts])


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=256):
    """L2-normalised CLIP text embeddings, as a float32 array (n, dim)."""
    out = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True).to(device)
        emb = model.get_text_features(**tokens).pooler_output
        out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu())
    return torch.cat(out).numpy()


def cosine_rows(a, b):
    """Row-wise cosine similarity of two L2-normalised arrays."""
    return np.einsum("ij,ij->i", a, b)
