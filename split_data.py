import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

GT_PATH = os.path.join(DATA_DIR, "ISIC_2024_Training_GroundTruth.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "ISIC_2024_Training_Input")

OUT_TRAIN = os.path.join(DATA_DIR, "train.csv")
OUT_VAL   = os.path.join(DATA_DIR, "val.csv")
OUT_TEST  = os.path.join(DATA_DIR, "test.csv")

RANDOM_STATE = 42

def main():
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Không thấy groundtruth: {GT_PATH}")
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Không thấy folder ảnh: {IMAGES_DIR}")

    df = pd.read_csv(GT_PATH)

    # CSV của m có: isic_id, malignant
    if "isic_id" not in df.columns:
        raise KeyError(f"CSV thiếu cột 'isic_id'. Hiện có: {list(df.columns)}")
    if "malignant" not in df.columns:
        raise KeyError(f"CSV thiếu cột 'malignant'. Hiện có: {list(df.columns)}")

    # dùng malignant làm nhãn (0/1)
    df["target"] = df["malignant"].astype(int)

    # build path ảnh
    df["image_path"] = df["isic_id"].astype(str).apply(
        lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg")
    )

    # loại bỏ dòng thiếu ảnh (nếu có)
    exists = df["image_path"].apply(os.path.exists)
    missing = (~exists).sum()
    if missing > 0:
        print(f"[WARN] {missing} dòng có image_path không tồn tại -> sẽ loại bỏ.")
        df = df[exists].copy()

    # split 70/15/15, stratify theo target
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["target"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["target"], random_state=RANDOM_STATE
    )

    # lưu
    train_df[["image_path", "target"]].to_csv(OUT_TRAIN, index=False)
    val_df[["image_path", "target"]].to_csv(OUT_VAL, index=False)
    test_df[["image_path", "target"]].to_csv(OUT_TEST, index=False)

    print("=== DONE ===")
    print("Saved:", OUT_TRAIN)
    print("Saved:", OUT_VAL)
    print("Saved:", OUT_TEST)
    print("train:", len(train_df), train_df["target"].value_counts().to_dict())
    print("val  :", len(val_df),   val_df["target"].value_counts().to_dict())
    print("test :", len(test_df),  test_df["target"].value_counts().to_dict())

if __name__ == "__main__":
    main()
