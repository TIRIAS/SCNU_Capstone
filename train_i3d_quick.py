# train_i3d_quick.py
import os, math, glob, argparse, random, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from einops import rearrange
from decord import VideoReader, cpu

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def list_videos(root):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
    paths, labels = [], []
    exts = (".mp4",".avi",".mov",".mkv",".wmv",".m4v")
    for i,c in enumerate(classes):
        for p in glob.glob(os.path.join(root,c,"**","*"), recursive=True):
            if p.lower().endswith(exts): paths.append(p); labels.append(i)
    return paths, labels, classes

def sample_frames(num_frames, T):
    if num_frames <= 0: return [0]*T
    idxs = np.linspace(0, max(0, num_frames-1), T).astype(int)
    return idxs.tolist()

def load_clip(path, T=16, size=224):
    vr = VideoReader(path, ctx=cpu(0))
    idxs = sample_frames(len(vr), T)
    frames = vr.get_batch(idxs).asnumpy()  # [T,H,W,3] uint8
    # Resize short side to 'size', center crop square, then to tensor [3,T,H,W]
    th, tw = frames.shape[1], frames.shape[2]
    # convert to torch for ops
    x = torch.from_numpy(frames)  # [T,H,W,3]
    # to CHW per frame
    x = x.permute(0,3,1,2).float() / 255.0  # [T,3,H,W]
    # resize keeping aspect
    resized = []
    for f in x:
        h,w = f.shape[-2:]
        if h < w:
            f = TF.resize(f, size=[size, int(w*size/h)])
        else:
            f = TF.resize(f, size=[int(h*size/w), size])
        resized.append(f)
    x = torch.stack(resized,0)  # [T,3,h',w']
    # center crop
    x = TF.center_crop(rearrange(x, "t c h w -> c t h w"), [size,size]) # [3,T,H,W]
    return x.contiguous()

class VideoFolder(Dataset):
    def __init__(self, root, T=16, size=224):
        self.paths, self.labels, self.classes = list_videos(root)
        assert self.paths, f"No videos found in {root}"
        self.T, self.size = T, size
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = load_clip(self.paths[i], self.T, self.size)  # [3,T,H,W]
        y = self.labels[i]
        return x, y

class I3DLite(nn.Module):
    # 간단한 Inflated 3D CNN (Conv3d stack) — I3D 입력/출력 규약 동일
    def __init__(self, num_classes=2, width=64):
        super().__init__()
        def block(cin, cout, k=3, s=(1,2,2), p=1):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True)
            )
        self.stem = block(3, width, k=7, s=(1,2,2), p=3)
        self.layer1 = nn.Sequential(block(width, width, s=(1,1,1)),
                                    block(width, width, s=(1,1,1)))
        self.down1  = block(width, width*2, s=(2,2,2))
        self.layer2 = nn.Sequential(block(width*2, width*2, s=(1,1,1)),
                                    block(width*2, width*2, s=(1,1,1)))
        self.down2  = block(width*2, width*4, s=(2,2,2))
        self.layer3 = nn.Sequential(block(width*4, width*4, s=(1,1,1)),
                                    block(width*4, width*4, s=(1,1,1)))
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.head   = nn.Linear(width*4, num_classes)
    def forward(self, x):  # x: [B,3,T,H,W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="dataset root having train/ and val/ subdirs")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = VideoFolder(os.path.join(args.root,"train"), T=args.frames, size=args.size)
    val_ds   = VideoFolder(os.path.join(args.root,"val"),   T=args.frames, size=args.size)
    num_classes = len(set(val_ds.labels + train_ds.labels))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=False)

    model = I3DLite(num_classes=num_classes, width=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    best = math.inf
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Train e{epoch}")
        for x,y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
            for x,y in val_dl:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = crit(logits, y)
                val_loss += loss.item() * y.size(0)
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        val_loss /= max(1,total)
        val_acc = correct / max(1,total)
        print(f"[epoch {epoch}] val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save({"model":model.state_dict(),
                        "classes": sorted(set(train_ds.labels+val_ds.labels))},
                       "i3d_lite_best.pt")
            print(">> checkpoint saved: i3d_lite_best.pt")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
