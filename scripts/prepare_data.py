# scripts/prepare_data.py
"""
Converts the ACL IMDB folder-of-files dataset into clean CSVs:
  data/train.csv  (from train/pos + train/neg minus validation portion)
  data/valid.csv  (stratified 20% of train)
  data/test.csv   (from test/pos + test/neg)
Optional:
  data/unsup.csv  (from train/unsup) if --with-unsup is passed

Columns:
  review  (str) : the text content
  label   (int) : 1 for positive, 0 for negative (NOT present in unsup.csv)
"""

from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_IMDB_ROOT = Path("/Users/sayalisawant/Downloads/aclImdb")

def load_split_dir(split_dir: Path) -> pd.DataFrame:
    """
    Reads all .txt files from split_dir/pos and split_dir/neg
    Returns a DataFrame with columns ['review', 'label'] (1=pos, 0=neg)
    """
    rows = []
    for label_name, label_val in [("pos", 1), ("neg", 0)]:
        p = split_dir / label_name
        assert p.exists(), f"Missing folder: {p}"
        for txt_fp in p.glob("*.txt"):
            text = txt_fp.read_text(encoding="utf-8", errors="ignore")
            rows.append({"review": text, "label": label_val})
    df = pd.DataFrame(rows)
    return df

def load_unsup_dir(unsup_dir: Path) -> pd.DataFrame:
    """
    Reads all .txt files from unsup_dir, returns DataFrame ['review'] (no label)
    """
    rows = []
    if unsup_dir.exists():
        for txt_fp in unsup_dir.glob("*.txt"):
            text = txt_fp.read_text(encoding="utf-8", errors="ignore")
            rows.append({"review": text})
    df = pd.DataFrame(rows)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_IMDB_ROOT),
                    help="Path to aclImdb root (contains train/ and test/)")
    ap.add_argument("--valid-size", type=float, default=0.2,
                    help="Fraction of TRAIN to reserve for validation (stratified)")
    ap.add_argument("--with-unsup", action="store_true",
                    help="Also create data/unsup.csv from train/unsup")
    args = ap.parse_args()

    imdb_root = Path(args.root)
    assert (imdb_root / "train").exists(), f"train/ not found in {imdb_root}"
    assert (imdb_root / "test").exists(), f"test/ not found in {imdb_root}"

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Load TRAIN labeled
    print("Loading TRAIN (pos/neg)...")
    df_train_full = load_split_dir(imdb_root / "train")
    print(f"  TRAIN total: {len(df_train_full):,} rows "
          f"(pos={df_train_full['label'].sum():,}, neg={(len(df_train_full)-df_train_full['label'].sum()):,})")

    # 2) Stratified split → TRAIN/VALID
    print(f"Creating stratified VALID split (size={args.valid_size:.2f})...")
    tr_df, val_df = train_test_split(
        df_train_full,
        test_size=args.valid_size,
        random_state=42,
        stratify=df_train_full["label"]
    )
    print(f"  TRAIN rows: {len(tr_df):,} | VALID rows: {len(val_df):,}")

    # 3) Load TEST labeled
    print("Loading TEST (pos/neg)...")
    df_test = load_split_dir(imdb_root / "test")
    print(f"  TEST total: {len(df_test):,} rows "
          f"(pos={df_test['label'].sum():,}, neg={(len(df_test)-df_test['label'].sum()):,})")

    # 4) Save CSVs
    tr_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "valid.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)
    print("Saved: data/train.csv, data/valid.csv, data/test.csv")

    # 5) Optional: UNSUP
    if args.with_unsup:
        print("Loading UNSUP (unlabeled)...")
        df_unsup = load_unsup_dir(imdb_root / "train" / "unsup")
        print(f"  UNSUP rows: {len(df_unsup):,}")
        df_unsup.to_csv(out_dir / "unsup.csv", index=False)
        print("Saved: data/unsup.csv")

    # 6) Sanity peek
    print("\nHead of train.csv:")
    print(tr_df.head(2))
    print("\nHead of valid.csv:")
    print(val_df.head(2))
    print("\nHead of test.csv:")
    print(df_test.head(2))

if __name__ == "__main__":
    main()
