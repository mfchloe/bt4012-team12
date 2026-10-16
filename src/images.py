"""Load DGM4 images straight from the zipped Hugging Face archives."""

import zipfile
from functools import lru_cache
from io import BytesIO

from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "rshaojimmy/DGM4"


def archive_for(image_path):
    # "DGM4/manipulation/infoswap/x.jpg" -> "manipulation/infoswap.zip"
    path_parts = image_path.removeprefix("DGM4/").split("/")
    return "/".join(path_parts[:2]) + ".zip"


@lru_cache(maxsize=None)
def open_archive(archive_name):
    archive_path = hf_hub_download(REPO_ID, archive_name, repo_type="dataset")
    archive = zipfile.ZipFile(archive_path)
    # Index members by path without the archive's top-level folder, so lookups are O(1)
    index = {name.split("/", 1)[-1]: name for name in archive.namelist()}
    return archive, index


def load_image(image_path, convert=True):
    """Return the image as RGB. With convert=False, return the decoded file as-is, which keeps JPEG
    metadata such as the quantisation tables."""
    archive_name = archive_for(image_path)
    archive, index = open_archive(archive_name)
    target_name = "/".join(image_path.removeprefix("DGM4/").split("/")[2:])
    if target_name not in index:
        raise FileNotFoundError(f"{target_name} was not found in {archive_name}")
    image = Image.open(BytesIO(archive.read(index[target_name])))
    return image.convert("RGB") if convert else image
