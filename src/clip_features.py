"""CLIP embeddings for DGM4 images, face crops and captions.

Images are loaded and preprocessed in DataLoader worker processes (so the GPU is not left waiting on
JPEG decoding), which is why the datasets live in a module rather than a notebook.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessorPil, CLIPModel, CLIPTokenizer

from src.images import load_image

CLIP_NAME = "openai/clip-vit-large-patch14"
FACE_MARGIN = 0.2  # enlarge each MTCNN box by this fraction on every side, so the crop includes hair and jaw


def best_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(device):
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    return model, CLIPImageProcessorPil.from_pretrained(CLIP_NAME), CLIPTokenizer.from_pretrained(CLIP_NAME)


def crop_face(image, box, margin=FACE_MARGIN):
    width, height = image.size
    x1, y1, x2, y2 = (int(v) for v in box)
    pad_x, pad_y = int(margin * (x2 - x1)), int(margin * (y2 - y1))
    return image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)))


class ImageDataset(Dataset):
    """Whole images (box=None) or face crops, preprocessed for CLIP."""

    def __init__(self, image_paths, boxes=None):
        self.image_paths = list(image_paths)
        self.boxes = None if boxes is None else list(boxes)
        self.processor = CLIPImageProcessorPil.from_pretrained(CLIP_NAME)

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
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        persistent_workers=num_workers > 0)
    out = []
    for pixels in loader:
        emb = model.get_image_features(pixel_values=pixels.to(device)).pooler_output
        out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu())
    return torch.cat(out).numpy()


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
