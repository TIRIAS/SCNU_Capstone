
# -*- coding: utf-8 -*-
r"""
train_binary_assault_v3_earlystop.py

- v2에서 추가/변경
  * Early Stopping: 검증 F1이 개선되지 않으면 조기 종료
    - --early_stop_patience: 개선 없는 에폭 허용 수(기본 3)
    - --early_stop_delta   : 개선으로 간주할 최소 F1 향상치(기본 0.005)
  * 종료 시 베스트 체크포인트 로드 후 테스트 평가

기타
  * 균형 샘플러, class-weighted CE / --focal, 임계값 튜닝, 혼동행렬 등 v2 기능 유지
"""

import time, argparse, random, math
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
except Exception:
    from torchvision.models.video import r3d_18
    R3D_18_Weights = None

# ======== 기본 설정 ========
DEF_ROOT = Path(r"D:\CCTV\CCTV\sample_dataset\yowo")
FRAMES_PER_CLIP = 16
SHORT_SIDE = 160
CROP_SIZE  = 160

MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

# AMP API 호환
try:
    from torch.amp import autocast as autocast_amp, GradScaler as GradScalerAmp
    def autocast_ctx(cuda_enabled): 
        return autocast_amp('cuda', enabled=cuda_enabled, dtype=torch.float16)
    def make_scaler(cuda_enabled): 
        return GradScalerAmp('cuda', enabled=cuda_enabled)
except Exception:
    from torch.cuda.amp import autocast as autocast_amp, GradScaler as GradScalerAmp
    def autocast_ctx(cuda_enabled): 
        return autocast_amp(enabled=cuda_enabled)
    def make_scaler(cuda_enabled): 
        return GradScalerAmp(enabled=cuda_enabled)

# ================= Dataset =================
def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_short_side(img, short_side:int):
    if short_side <= 0: return img
    h, w = img.shape[:2]
    if h < w:
        new_h = short_side
        new_w = int(round(w * (short_side / h)))
    else:
        new_w = short_side
        new_h = int(round(h * (short_side / w)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def center_crop(img, size:int):
    h, w = img.shape[:2]
    th, tw = size, size
    y1 = max(0, (h - th) // 2)
    x1 = max(0, (w - tw) // 2)
    return img[y1:y1+th, x1:x1+tw]

def random_horizontal_flip_clip(frames:list, p=0.5):
    if random.random() < p:
        return [cv2.flip(f, 1) for f in frames]
    return frames

def norm_to_tensor(frames:list):
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # T,H,W,C float32
    arr = (arr - MEAN[None,None,None,:]) / STD[None,None,None,:]
    arr = np.transpose(arr, (3,0,1,2))  # C,T,H,W
    return torch.from_numpy(arr)  # float32

class ClipListDataset(Dataset):
    def __init__(self, yowo_dir:Path, list_file:Path, train:bool):
        self.yowo_dir = yowo_dir
        self.train = train
        items = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                rel, lab = line.rsplit(" ", 1)
                items.append((rel, int(lab)))
        self.items = items

    def __len__(self): return len(self.items)

    def _load_clip(self, clip_dir:Path):
        frames = []
        for i in range(1, FRAMES_PER_CLIP+1):
            fp = clip_dir / f"frame_{i:05d}.jpg"
            frames.append(imread_rgb(fp))
        frames = [resize_short_side(f, SHORT_SIDE) for f in frames]
        if self.train:
            frames = random_horizontal_flip_clip(frames, p=0.5)
        frames = [center_crop(f, CROP_SIZE) for f in frames]
        return frames

    def __getitem__(self, idx):
        rel, lab = self.items[idx]
        clip_dir = self.yowo_dir / rel
        frames = self._load_clip(clip_dir)
        x = norm_to_tensor(frames)  # float32
        y = torch.tensor(lab, dtype=torch.long)
        return x, y

# =============== Utils ===============
def collate_fn(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # N,C,T,H,W
    y = torch.stack(ys, dim=0)
    return x, y

def accuracy_from_pred(pred, y):
    return (pred == y).float().mean().item()

def f1_pr_from_pred(pred, y, positive=1):
    with torch.no_grad():
        pred = pred.cpu()
        y = y.cpu()
        tp = int(((pred==positive) & (y==positive)).sum())
        fp = int(((pred==positive) & (y!=positive)).sum())
        fn = int(((pred!=positive) & (y==positive)).sum())
        precision = tp / (tp+fp+1e-8)
        recall = tp / (tp+fn+1e-8)
        f1 = 2*precision*recall/(precision+recall+1e-8)
        return f1, precision, recall, tp, fp, fn

def softmax_prob(logits):
    return torch.softmax(logits, dim=1)[:,1]

# =============== Model/Loss ===============
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')
    def forward(self, logits, target):
        ce = self.ce(logits, target)  # (N,)
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def get_model(num_classes=2, try_pretrained=True):
    used_pretrained = False
    if try_pretrained and R3D_18_Weights is not None:
        try:
            weights = R3D_18_Weights.KINETICS400_V1
            model = r3d_18(weights=weights)
            used_pretrained = True
        except Exception:
            model = r3d_18(weights=None)
    else:
        model = r3d_18(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model, used_pretrained

def run_epoch(model, loader, optimizer, scaler, device, criterion, train=True):
    model.train(train)
    total_loss, total_acc, total_f1 = 0.0, 0.0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)

        with autocast_ctx(device.type=='cuda'):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        pred = logits.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy_from_pred(pred, y) * x.size(0)
        f1, p, r, *_ = f1_pr_from_pred(pred, y)
        total_f1  += f1 * x.size(0)

    n = len(loader.dataset)
    return total_loss/n, total_acc/n, total_f1/n

def evaluate_logits(model, loader, device):
    model.eval()
    all_probs = []
    all_y = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            logits = model(x)
            probs = softmax_prob(logits)
            all_probs.append(probs.cpu())
            all_y.append(y.cpu())
    all_probs = torch.cat(all_probs, dim=0)
    all_y = torch.cat(all_y, dim=0)
    return all_probs, all_y

def find_best_threshold(probs, y):
    # y: 0/1 tensor
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 41):
        pred = (probs.numpy() >= t).astype(np.int64)
        pred = torch.from_numpy(pred)
        f1, p, r, *_ = f1_pr_from_pred(pred, y, positive=1)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1

def make_sampler(dataset:Dataset):
    # 클래스 비율 기반 가중치 → 클래스 균형 샘플러
    labels = [lab for _, lab in dataset.items]
    cnt = Counter(labels)
    total = len(labels)
    class_weight = {c: total/(len(cnt)*cnt[c]) for c in cnt}  # inverse freq
    weights = [class_weight[lab] for lab in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
    return sampler, class_weight

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yowo_dir", type=str, default=str(DEF_ROOT))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--output", type=str, default="runs_assault_v3_earlystop")
    ap.add_argument("--focal", action="store_true", help="Focal Loss 사용")
    ap.add_argument("--no_pretrained", action="store_true", help="사전학습 가중치 미사용")
    ap.add_argument("--early_stop_patience", type=int, default=3, help="개선 없는 에폭 허용 수")
    ap.add_argument("--early_stop_delta", type=float, default=0.005, help="개선으로 간주할 최소 F1 향상치")
    args = ap.parse_args()

    yowo_dir = Path(args.yowo_dir)
    train_list = yowo_dir / "train_list.txt"
    val_list   = yowo_dir / "val_list.txt"
    test_list  = yowo_dir / "test_list.txt"

    ds_tr = ClipListDataset(yowo_dir, train_list, train=True)
    ds_va = ClipListDataset(yowo_dir, val_list,   train=False)
    ds_te = ClipListDataset(yowo_dir, test_list,  train=False)

    # 분포 출력
    def dist(ds):
        cnt = Counter([lab for _,lab in ds.items])
        return dict(cnt), len(ds)
    dtr, ntr = dist(ds_tr); dva, nva = dist(ds_va); dte, nte = dist(ds_te)
    print(f"[DATA] train={ntr} {dtr} | val={nva} {dva} | test={nte} {dte}")

    # Sampler & class weights
    sampler, class_weight = make_sampler(ds_tr)
    print(f"[WEIGHT] class_weight(train) = {class_weight}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,      num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,      num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEV] device={device}")

    model, used_pretrained = get_model(num_classes=2, try_pretrained=not args.no_pretrained)
    print(f"[MODEL] pretrained={used_pretrained}")
    model = model.to(device)

    alpha = torch.tensor([class_weight.get(0,1.0), class_weight.get(1,1.0)], dtype=torch.float32, device=device)
    if args.focal:
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        print("[LOSS] FocalLoss(gamma=2) with class alpha")
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha)
        print("[LOSS] CrossEntropyLoss with class weights")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_scaler(device.type=='cuda')

    best_va_f1, best_info, best_path = -1.0, None, None
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Early Stopping 상태 ----
    patience = args.early_stop_patience
    delta    = args.early_stop_delta
    no_improve = 0

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, dl_tr, optimizer, scaler, device, criterion, train=True)
        # threshold-free val logits
        va_probs, va_y = evaluate_logits(model, dl_va, device)
        # tune threshold on val
        th, va_f1 = find_best_threshold(va_probs, va_y)
        va_pred = (va_probs.numpy() >= th).astype(np.int64)
        va_pred = torch.from_numpy(va_pred)
        va_acc = accuracy_from_pred(va_pred, va_y)
        f1, p, r, *_ = f1_pr_from_pred(va_pred, va_y)
        scheduler.step()
        dt = time.time()-t0
        print(f"[E{ep:02d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
              f"val@th={th:.2f} acc {va_acc:.3f} f1 {f1:.3f} (P {p:.3f} R {r:.3f}) | {dt:.1f}s")

        # Save best
        improved = (f1 - best_va_f1) > delta
        if improved:
            best_va_f1 = f1
            best_info = {"ep":ep, "th":th, "f1":f1, "p":p, "r":r}
            best_path = out_dir / f"best_ep{ep:02d}_f1{f1:.3f}_th{th:.2f}.pt"
            torch.save({"ep":ep, "model":model.state_dict(), "th":th, "args":vars(args)}, best_path)
            print(f"[CKPT] saved {best_path}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"[EARLY] no improvement ({no_improve}/{patience})")

        # Early stop check
        if no_improve >= patience:
            print(f"[EARLY STOP] Stop at epoch {ep} (best val F1={best_va_f1:.3f})")
            break

    # ---- 테스트 평가 (최적 임계값 사용) ----
    if best_info and best_path:
        th = best_info["th"]
        print(f"[BEST] ep={best_info['ep']} th={th:.2f} val_f1={best_info['f1']:.3f} P={best_info['p']:.3f} R={best_info['r']:.3f}")
        # 안전을 위해 state_dict만 로드
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    else:
        th = 0.5
        print("[BEST] no improvement recorded; using th=0.50")

    # 테스트
    te_probs, te_y = evaluate_logits(model, dl_te, device)
    te_pred = (te_probs.numpy() >= th).astype(np.int64)
    te_pred = torch.from_numpy(te_pred)
    te_acc = accuracy_from_pred(te_pred, te_y)
    def f1_pr_from_arrays(pred_arr, y_tensor):
        pred_t = torch.from_numpy(pred_arr) if not torch.is_tensor(pred_arr) else pred_arr
        return f1_pr_from_pred(pred_t, y_tensor)
    te_f1, te_p, te_r, tp, fp, fn = f1_pr_from_arrays(te_pred, te_y)
    tn = int(len(te_y) - tp - fp - fn)
    print(f"[TEST] th={th:.2f} acc {te_acc:.3f} f1 {te_f1:.3f} P {te_p:.3f} R {te_r:.3f} | cm: TP {tp} TN {tn} FP {fp} FN {fn}")

    with open(out_dir / "test_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"th {th:.2f} acc {te_acc:.3f} f1 {te_f1:.3f} p {te_p:.3f} r {te_r:.3f} tp {tp} tn {tn} fp {fp} fn {fn}\n")

if __name__ == "__main__":
    main()
