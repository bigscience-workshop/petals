import argparse
from datetime import datetime

from huggingface_hub import delete_repo, list_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove old testing models from HF hub")
    parser.add_argument("--author", type=str, default="bloom-testing", help="auth token for from_pretrained")
    parser.add_argument("--seconds_since_last_updated", type=int, default=7 * 24 * 60 * 60)
    parser.add_argument("--use_auth_token", type=str, default=None, help="auth token for from_pretrained")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    for model in list_models(author=args.author, full=True):
        last_modified = datetime.strptime(model.lastModified, "%Y-%m-%dT%H:%M:%S.%fZ")

        if model.modelId.endswith("-main") or "/test-" not in model.modelId:
            continue  # remove only test models

        if (datetime.now() - last_modified).total_seconds() > args.seconds_since_last_updated:
            if args.dry_run:
                print(f"{model.modelId} can be deleted")
            else:
                delete_repo(token=args.use_auth_token, name=model.modelId, organization=args.author)
