import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# CONFIG
# =========================
# Đường dẫn gốc của bạn
BASE_DIR = r"E:\24022314\Học máy\Machine_Learning_project\data"

# Cấu hình đường dẫn chi tiết
TRAIN_CSV = os.path.join(BASE_DIR, "data", "train_ref10.csv")
VAL_CSV   = os.path.join(BASE_DIR, "val.csv")
IMG_DIR   = os.path.join(BASE_DIR, "ISIC_2024_Training_Input") # Thư mục chứa ảnh

# Tên cột trong CSV chứa tên file ảnh (thường là isic_id)
ID_COL    = "isic_id" 
LABEL_COL = "target"

MODEL_NAME = "swin_tiny_patch4_window7_224"
IMG_SIZE = 224          # Swin tiny chuẩn chạy 224
BATCH_SIZE = 16         # Nếu GPU 4GB VRAM thì giảm xuống 8
EPOCHS = 3
LR = 2e-4 
WEIGHT_DECAY = 1e-4

# QUAN TRỌNG: Để 0 để sửa lỗi WinError trên Windows
NUM_WORKERS = 0  

SEED = 42

# Imbalance handling
USE_SAMPLER = True      
POS_AUG_ONLY = True     

# Loss
USE_FOCAL = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.90      

# Train tricks
USE_AMP = True
GRAD_CLIP = 1.0
USE_TTA_EVAL = True     

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
# =========================


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.90):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        targets = targets.long()
        # logits: [N, 2] -> log_softmax -> [N, 2]
        logp = torch.log_softmax(logits, dim=1)
        p = torch.softmax(logits, dim=1)

        # Lấy xác suất đúng class (pt)
        pt = p[torch.arange(logits.size(0), device=logits.device), targets]
        logpt = logp[torch.arange(logits.size(0), device=logits.device), targets]

        # Tính alpha_t
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1.0 - self.alpha, device=logits.device)
        )

        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * logpt
        return loss.mean()


class ISICDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform_neg, transform_pos, id_col="isic_id", label_col="target"):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.id_col = id_col
        self.label_col = label_col
        self.t_neg = transform_neg
        self.t_pos = transform_pos
        
        # Kiểm tra cột
        if self.label_col not in self.df.columns:
            # Fallback: nếu không thấy cột target, thử lấy cột cuối cùng
            print(f"Lưu ý: Không thấy cột '{self.label_col}', dùng cột cuối cùng làm nhãn.")
            self.label_col = self.df.columns[-1]

        if self.id_col not in self.df.columns:
            # Fallback: lấy cột đầu tiên làm ID ảnh
            print(f"Lưu ý: Không thấy cột '{self.id_col}', dùng cột đầu tiên làm ID ảnh.")
            self.id_col = self.df.columns[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Lấy tên ảnh và ghép đường dẫn
        img_name = row[self.id_col]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Tạo ảnh đen nếu thiếu file để code không crash
            img = Image.new('RGB', (224, 224))
            
        y = int(row[self.label_col])

        if y == 1:
            img = self.t_pos(img)
        else:
            img = self.t_neg(img)

        return img, y


@torch.no_grad()
def evaluate_tta(model, loader, device, tta=True):
    model.eval()
    ys, probs = [], []

    def predict_prob(x):
        logits = model(x)
        return torch.softmax(logits, dim=1)[:, 1]

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        if not tta:
            p = predict_prob(x)
        else:
            # Test Time Augmentation (TTA)
            preds = []
            preds.append(predict_prob(x)) # Gốc
            preds.append(predict_prob(torch.flip(x, dims=[3]))) # Lật ngang
            preds.append(predict_prob(torch.flip(x, dims=[2]))) # Lật dọc
            p = torch.stack(preds, dim=0).mean(dim=0)

        probs.extend(p.detach().cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())

    ys = np.asarray(ys, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    
    # Tránh lỗi nếu chỉ có 1 class trong batch validation
    if len(np.unique(ys)) < 2:
        return 0.0, 0.0

    pr_auc = average_precision_score(ys, probs)
    roc_auc = roc_auc_score(ys, probs)
    return pr_auc, roc_auc


def main():
    # Fix lỗi multiprocessing trên Windows khi gọi num_workers > 0 (nếu có)
    # Dù đã set num_workers=0 nhưng cứ để dòng này cho chuẩn
    torch.backends.cudnn.benchmark = True 
    
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data: {BASE_DIR}")
    print(f"Workers: {NUM_WORKERS} (Set 0 to avoid WinError 1455)")

    # ========= Transforms =========
    train_neg = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    if POS_AUG_ONLY:
        train_pos = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
    else:
        train_pos = train_neg

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    # Load Dataset với đường dẫn ảnh riêng
    print("Loading datasets...")
    train_ds = ISICDataset(TRAIN_CSV, IMG_DIR, transform_neg=train_neg, transform_pos=train_pos, id_col=ID_COL, label_col=LABEL_COL)
    val_ds   = ISICDataset(VAL_CSV, IMG_DIR, transform_neg=val_tf, transform_pos=val_tf, id_col=ID_COL, label_col=LABEL_COL)

    # Đếm số lượng class để làm Sampler
    # Lưu ý: lấy cột label từ dataframe
    y_train = train_ds.df[train_ds.label_col].astype(int).values
    counts = np.bincount(y_train, minlength=2)
    neg, pos = int(counts[0]), int(counts[1])
    print(f"Train stats: Neg={neg}, Pos={pos}")

    # ========= Sampler =========
    sampler = None
    shuffle = True
    if USE_SAMPLER and pos > 0:
        print("Using WeightedRandomSampler...")
        class_w = 1.0 / np.maximum(counts, 1)
        sample_w = class_w[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).double(),
            num_samples=len(sample_w),
            replacement=True
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda")
    )

    # ========= Model =========
    print(f"Creating model: {MODEL_NAME}")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2)
    model.to(device)

    # ========= Loss =========
    if USE_FOCAL:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
        print("Loss: FocalLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Loss: CrossEntropy")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Cập nhật cú pháp AMP mới
    scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and device == "cuda"))

    best_pr = -1.0
    
    # ========= Train loop =========
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(USE_AMP and device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if GRAD_CLIP is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP) 

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Evaluate
        print("Evaluating...")
        pr_auc, roc_auc = evaluate_tta(model, val_loader, device, tta=USE_TTA_EVAL)
        
        print(f"RESULT Epoch {ep}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val PR-AUC: {pr_auc:.4f} (Max: {best_pr:.4f})")
        print(f"  Val ROC-AUC: {roc_auc:.4f}")

        if pr_auc > best_pr:
            best_pr = pr_auc
            save_path = os.path.join(OUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved to {save_path}")

    print(f"Done! Best PR-AUC: {best_pr:.4f}")

if __name__ == "__main__":
    main()