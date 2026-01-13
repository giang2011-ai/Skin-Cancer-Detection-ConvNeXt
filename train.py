import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from timm.utils import ModelEmaV2

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)

import matplotlib.pyplot as plt

from torch.amp import autocast, GradScaler  # new AMP API



# ======================
# CONFIG CHO SKIN CANCER (0.1%)
# ======================
DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train_ref10.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_ref10.csv")
LABEL_COL = "target"

MODEL_NAME = "convnext_tiny_scratch"  
IMG_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 20
LR = 5e-5
NUM_WORKERS = 4

SEEDS = [42, 3407, 2024]

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# UTILS
# ======================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float = 0.999):
    # ema = decay * ema + (1-decay) * model
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1.0 - decay)


class ImageCSVDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, label_col: str = "target"):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        y = float(row[self.label_col])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor([y], dtype=torch.float32)


# ======================
# BINARY MIXUP
# ======================
class BinaryMixup:
    """
    Mixup cho bài toán binary:
      - model output: 1 logit
      - loss: BCEWithLogitsLoss
      - y: float tensor shape (B,1) in {0,1}
    """
    def __init__(self, alpha=0.4, prob=0.7, label_smoothing=0.05):
        self.alpha = float(alpha)
        self.prob = float(prob)
        self.label_smoothing = float(label_smoothing)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        # label smoothing nhẹ 
        def smooth(t):
            if self.label_smoothing <= 0:
                return t
            return t * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.alpha <= 0 or random.random() > self.prob:
            return x, smooth(y)

        lam = float(np.random.beta(self.alpha, self.alpha))

        b = x.size(0)
        perm = torch.randperm(b, device=x.device)

        x2 = x[perm]
        y2 = y[perm]

        x_mix = x * lam + x2 * (1.0 - lam)
        y_mix = y * lam + y2 * (1.0 - lam)

        return x_mix, smooth(y_mix)


# ======================
# TTA predict (ONE MODEL)
# ======================
@torch.no_grad()
def predict_logits_tta(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,H,W) on device
    return: (B,1) logits (chưa sigmoid), đã TTA mean
    """
    model.eval()

    x_hflip = torch.flip(x, dims=[3])
    x_vflip = torch.flip(x, dims=[2])
    x_rot90 = torch.rot90(x, 1, [2, 3])

    x_tta = torch.cat([x, x_hflip, x_vflip, x_rot90], dim=0)  # (4B, C, H, W)
    logits = model(x_tta)  # (4B,1)

    bs = x.size(0)
    l1, l2, l3, l4 = torch.split(logits, bs, dim=0)
    return (l1 + l2 + l3 + l4) / 4.0  # (B,1)


@torch.no_grad()
def predict_proba_tta(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = predict_logits_tta(model, x)
    return torch.sigmoid(logits)


# ======================
# EVAL + COLLECT PROBS
# ======================
@torch.no_grad()
def evaluate_ensemble(models, loader, device):
    """
    Ensemble = mean(LOGITS) across models -> sigmoid -> prob
    """
    for m in models:
        m.eval()

    ys, probs = [], []

    for x, y in loader:
        x = x.to(device)

        logits_list = [predict_logits_tta(m, x) for m in models]  # list of (B,1)
        logits_avg = torch.stack(logits_list, dim=0).mean(dim=0)  # (B,1)
        p_avg = torch.sigmoid(logits_avg)

        probs.extend(p_avg.cpu().numpy().flatten().tolist())
        ys.extend(y.numpy().flatten().tolist())

    ys = np.array(ys)
    probs = np.array(probs)

    pr_auc = average_precision_score(ys, probs)
    try:
        roc_auc = roc_auc_score(ys, probs)
    except Exception:
        roc_auc = 0.5

    return pr_auc, roc_auc


@torch.no_grad()
def collect_probs_ensemble(models, loader, device):
    """
    Trả về y_true và y_score (prob) để vẽ PR curve.
    """
    for m in models:
        m.eval()

    ys, probs = [], []

    for x, y in loader:
        x = x.to(device)

        logits_list = [predict_logits_tta(m, x) for m in models]
        logits_avg = torch.stack(logits_list, dim=0).mean(dim=0)  # (B,1)
        p_avg = torch.sigmoid(logits_avg)

        probs.extend(p_avg.cpu().numpy().flatten().tolist())
        ys.extend(y.numpy().flatten().tolist())

    return np.array(ys), np.array(probs)


def plot_pr_curve(y_true, y_score, out_path, title="Precision-Recall Curve"):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc_val = auc(rec, prec)

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (PR-AUC = {pr_auc_val:.4f})")
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return pr_auc_val


def plot_pr_auc_by_epoch(hist_paths, out_path):
    """
    Vẽ PR-AUC theo epoch (hình phụ). Vẽ 3 seed thành 3 đường.
    """
    plt.figure()
    for hp in hist_paths:
        df = pd.read_csv(hp)
        label = os.path.basename(hp).replace(".csv", "")
        plt.plot(df["epoch"], df["pr_auc"], label=label)

    plt.xlabel("Epoch")
    plt.ylabel("PR-AUC (val, EMA)")
    plt.title("PR-AUC theo epoch (hình phụ)")
    plt.grid(True)
    plt.legend(fontsize=7)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ======================
# TRAIN ONE SEED (LOG HISTORY)
# ======================
def train_one_seed(seed: int, train_loader, val_loader, device: str):
    seed_all(seed)
    print(f"\n========== TRAIN SEED = {seed} ==========")

    mixup_fn = BinaryMixup(alpha=0.4, prob=0.7, label_smoothing=0.05)

    model = create_model(
        model_name="convnext_tiny",
        pretrained=False,     
        num_classes=1,
        drop_path_rate=0.1
    ).to(device)

    ema_decay = 0.999
    model_ema = AveragedModel(model).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    scaler = GradScaler('cuda') if device == "cuda" else None

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1
    )

    best_pr = 0.0
    best_path = os.path.join(OUT_DIR, f"best_seed{seed}_{MODEL_NAME}.pt")

    history = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Seed {seed}] Epoch {ep}/{EPOCHS}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            x_mix, y_mix = mixup_fn(x, y)

            optimizer.zero_grad(set_to_none=True)

            if device == "cuda":
                with autocast('cuda'):
                    logits = model(x_mix)
                    loss = criterion(logits, y_mix)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_mix)
                loss = criterion(logits, y_mix)
                loss.backward()
                optimizer.step()

            ema_update(model_ema, model, decay=ema_decay)

            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pr_auc, roc_auc = evaluate_ensemble([model_ema], val_loader, device)
        print(f" >> [VAL EMA][Seed {seed}] PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

        history.append({"epoch": ep, "pr_auc": pr_auc, "roc_auc": roc_auc})

        if pr_auc > best_pr:
            best_pr = pr_auc
            torch.save(model_ema.state_dict(), best_path)
            print(f"Saved Best EMA Model: {best_path}")

    hist_path = os.path.join(OUT_DIR, f"history_seed{seed}_{MODEL_NAME}.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"[Seed {seed}] Saved history: {hist_path}")

    print(f"[Seed {seed}] Done. Best PR-AUC: {best_pr:.4f}")
    return best_path, hist_path


# ======================
# LOAD ENSEMBLE
# ======================
def load_models_from_paths(paths, device: str):
    models = []
    for p in paths:
        m = create_model(
            model_name="convnext_tiny",
            pretrained=False,
            num_classes=1,
            drop_path_rate=0.0
        )
        sd = torch.load(p, map_location="cpu")
        m.load_state_dict(sd, strict=True)
        m.to(device)
        m.eval()
        models.append(m)
    return models


# ======================
# MAIN
# ======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {MODEL_NAME} | Ensemble Seeds: {SEEDS}")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageCSVDataset(TRAIN_CSV, transform=train_tf, label_col=LABEL_COL)
    val_ds   = ImageCSVDataset(VAL_CSV,   transform=val_tf,   label_col=LABEL_COL)

    # WeightedRandomSampler
    y_train = train_ds.df[LABEL_COL].values.astype(int)
    class_counts = np.bincount(y_train, minlength=2)
    class_counts = np.maximum(class_counts, 1)
    weight_per_class = 1.0 / class_counts
    samples_weight = torch.from_numpy(weight_per_class[y_train]).double()

    sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 1) TRAIN 3 SEEDS + LOG HISTORY
    best_paths = []
    hist_paths = []
    for seed in SEEDS:
        best_path, hist_path = train_one_seed(seed, train_loader, val_loader, device)
        best_paths.append(best_path)
        hist_paths.append(hist_path)

    print("\nAll best checkpoints:")
    for p in best_paths:
        print(" -", p)

    # 2) LOAD ENSEMBLE & EVAL FINAL
    models = load_models_from_paths(best_paths, device)
    pr_auc, roc_auc = evaluate_ensemble(models, val_loader, device)

    print(f"\n========== FINAL ENSEMBLE (mean probs) ==========")
    print(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

    # 3) VẼ HÌNH CHÍNH: Precision-Recall curve (ensemble)
    y_true, y_score = collect_probs_ensemble(models, val_loader, device)
    pr_curve_path = os.path.join(OUT_DIR, "pr_curve_ensemble.png")
    pr_auc_curve = plot_pr_curve(y_true, y_score, pr_curve_path, title="Precision-Recall Curve (Ensemble)")
    print(f"Saved PR curve: {pr_curve_path} (PR-AUC from curve = {pr_auc_curve:.4f})")

    # 4) VẼ HÌNH PHỤ: PR-AUC theo epoch (val, EMA)
    pr_epoch_path = os.path.join(OUT_DIR, "pr_auc_by_epoch.png")
    plot_pr_auc_by_epoch(hist_paths, pr_epoch_path)
    print(f"Saved PR-AUC by epoch: {pr_epoch_path}")


if __name__ == "__main__":
    main()
