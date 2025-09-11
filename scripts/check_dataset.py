# scripts/check_dataset.py
from pathlib import Path

# ⬇️ CHANGE THIS to your actual path for the ACL IMDB dataset root
# It should contain folders: train/pos, train/neg, train/unsup, test/pos, test/neg
IMDB_ROOT = Path("/Users/sayalisawant/Downloads/aclImdb")  # e.g., "/Users/yourname/Downloads/aclImdb"

def count_txt(p: Path) -> int:
    return len(list(p.glob("*.txt"))) if p.exists() else 0

def main():
    assert IMDB_ROOT.exists(), f"Not found: {IMDB_ROOT}"
    print(f"Dataset root: {IMDB_ROOT}")
    for split in ["train", "test"]:
        for label in ["pos", "neg"]:
            p = IMDB_ROOT / split / label
            print(f"{split}/{label}: {count_txt(p):,} files")
    unsup = IMDB_ROOT / "train" / "unsup"
    print(f"train/unsup: {count_txt(unsup):,} files (unlabeled)")

if __name__ == "__main__":
    main()
