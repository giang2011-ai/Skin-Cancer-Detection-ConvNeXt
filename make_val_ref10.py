import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SRC_VAL = os.path.join(DATA_DIR, "val.csv")
OUT_VAL = os.path.join(DATA_DIR, "val_ref10.csv")

LABEL_COL = "target"
POS_RATE = 0.10
SEED = 42


def main():
    print("SRC_VAL:", SRC_VAL)
    if not os.path.exists(SRC_VAL):
        raise FileNotFoundError(f"Không thấy {SRC_VAL}")

    df = pd.read_csv(SRC_VAL)
    if LABEL_COL not in df.columns:
        raise KeyError(f"val.csv thiếu cột '{LABEL_COL}'. Hiện có: {list(df.columns)}")

    pos = df[df[LABEL_COL] == 1]
    neg = df[df[LABEL_COL] == 0]

    n_pos = len(pos)
    if n_pos == 0:
        raise ValueError("Không có sample positive trong val.csv")

    # pos / (pos + neg_need) = POS_RATE
    neg_need = int(round(n_pos * (1 - POS_RATE) / POS_RATE))
    neg_need = min(neg_need, len(neg))

    out = pd.concat(
        [pos, neg.sample(n=neg_need, random_state=SEED)],
        ignore_index=True
    )
    out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)

    out.to_csv(OUT_VAL, index=False)

    counts = out[LABEL_COL].value_counts().to_dict()
    rate = counts.get(1, 0) / len(out)

    print("✅ Saved:", OUT_VAL)
    print("Total:", len(out))
    print("Counts:", counts)
    print("Positive rate:", round(rate * 100, 2), "%")


if __name__ == "__main__":
    main()
