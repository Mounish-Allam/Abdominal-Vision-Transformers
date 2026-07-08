"""
Upload your trained .pth checkpoint and model card to a Hugging Face Model Hub repository.

Usage:
    python upload_weights.py \
        --weights model/Best_SwinDAF-CHAOS.pth \
        --repo    your-username/swin-daf-chaos-mri

You must be logged in first:
    huggingface-cli login
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ROOT = Path(__file__).resolve().parent
MODEL_CARD_PATH = REPO_ROOT / "model_card.md"


def upload(weights_path: str, repo_id: str, private: bool = False) -> None:
    api = HfApi()

    print(f"Creating repo '{repo_id}' (private={private}) if it doesn't exist ...")
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    filename = weights_path.split("/")[-1].split("\\")[-1]
    print(f"Uploading '{weights_path}' as '{filename}' ...")
    url = api.upload_file(
        path_or_fileobj=weights_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Weights uploaded: {url}")

    if MODEL_CARD_PATH.is_file():
        print(f"Uploading model card '{MODEL_CARD_PATH}' as 'README.md' ...")
        card_url = api.upload_file(
            path_or_fileobj=str(MODEL_CARD_PATH),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Model card uploaded: {card_url}")
    else:
        print(f"WARNING: no model card found at {MODEL_CARD_PATH}, skipping.")

    print(
        f"\nDone! Now set this as a Space environment variable:\n"
        f"  HF_MODEL_REPO_ID  = {repo_id}\n"
        f"  HF_MODEL_FILENAME = {filename}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Local path to .pth file")
    parser.add_argument("--repo",    required=True, help="HF repo id, e.g. username/model-name")
    parser.add_argument("--private", action="store_true", help="Make the model repo private")
    args = parser.parse_args()
    upload(args.weights, args.repo, args.private)
