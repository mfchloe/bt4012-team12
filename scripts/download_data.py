"""Download DGM4 from Hugging Face and unpack it into data/raw/DGM4.

Resulting layout matches the `image` paths in the metadata, so an image is at
data/raw/<image path>, e.g. data/raw/DGM4/manipulation/infoswap/995762-043201-infoswap.jpg

Usage:
    python scripts/download_data.py                     # everything (~10.7GB download, ~11GB on disk)
    python scripts/download_data.py --only infoswap bbc # metadata + selected archives
    python scripts/download_data.py --clear-cache       # also delete the zips from the HF cache afterwards
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, scan_cache_dir

REPO_ID = "rshaojimmy/DGM4"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATASET_DIR = RAW_DIR / "DGM4"

METADATA_FILES = ["metadata/train.json", "metadata/val.json", "metadata/test.json"]
ARCHIVES = [
    "origin/bbc.zip",
    "origin/guardian.zip",
    "origin/usa_today.zip",
    "origin/washington_post.zip",
    "manipulation/infoswap.zip",
    "manipulation/simswap.zip",
    "manipulation/HFGI.zip",
    "manipulation/StyleCLIP.zip",
]


def download(filename):
    # Goes through the HF cache, so files already downloaded are not fetched again
    return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset"))


def extract(archive_name):
    # "manipulation/infoswap.zip" contains "infoswap/...", so extract into "manipulation/"
    target_dir = DATASET_DIR / Path(archive_name).parent
    done_marker = target_dir / f".{Path(archive_name).stem}.done"
    if done_marker.exists():
        print(f"  already extracted: {archive_name}")
        return

    archive_path = download(archive_name)
    print(f"  extracting {archive_name} -> {target_dir.relative_to(RAW_DIR.parents[1])}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)
    done_marker.touch()


def clear_hf_cache():
    cache = scan_cache_dir()
    revisions = [
        revision.commit_hash
        for repo in cache.repos
        if repo.repo_id == REPO_ID
        for revision in repo.revisions
    ]
    strategy = cache.delete_revisions(*revisions)
    print(f"Freeing {strategy.expected_freed_size_str} from the Hugging Face cache")
    strategy.execute()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="archive names to fetch, e.g. infoswap bbc (default: all)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="delete the downloaded zips from the HF cache after extracting")
    args = parser.parse_args()

    archives = ARCHIVES
    if args.only:
        archives = [a for a in ARCHIVES if Path(a).stem in args.only]
        unknown = set(args.only) - {Path(a).stem for a in archives}
        if unknown:
            parser.error(f"unknown archive(s): {', '.join(sorted(unknown))}")

    print("Metadata")
    (DATASET_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    for filename in METADATA_FILES:
        shutil.copyfile(download(filename), DATASET_DIR / filename)
        print(f"  {filename}")

    print("Images")
    for archive_name in archives:
        extract(archive_name)

    if args.clear_cache:
        clear_hf_cache()

    print(f"Done. Data is in {DATASET_DIR}")


if __name__ == "__main__":
    main()
