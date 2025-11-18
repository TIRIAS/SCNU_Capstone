
import os, sys, csv, cv2, numpy as np, torch, warnings
from collections import deque
from pytorchvideo.models.hub import slowfast_r50
from ultralytics import YOLO

# ================== Config ==================
CKPT = r"D:\\large_data\\checkpoints\\slowfast_r50_4cls_balanced.pt"
YOLO_WEIGHTS = r"C:\\Users\\user\\runs\\detect\\train18\\weights\\best.pt"
OUT_DIR = r"D:\\large_data\\inference_out"
SAVE_VIDEO = True
SAVE_CSV = True

# Gating / ROI
PERSON_CONF = 0.60
MIN_AREA_RATIO = 0.005
ASPECT_MIN, ASPECT_MAX = 0.25, 0.90
NO_PERSON_CLEAR = 8
ROI_SCALE = 1.35
ROI_MIN_AREA_RATIO = 0.05
BG_MODE = "gray"  # or "blur"

# SlowFast sampling & norm
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")
MEAN = torch.tensor([0.45,0.45,0.45]).view(1,3,1,1,1)
STD  = torch.tensor([0.225,0.225,0.225]).view(1,3,1,1,1)

# ================== Utils ==================
def center_crop_224_rgb(bgr, size):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256.0 / max(1, min(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = max(0, (nh - size)//2), max(0, (nw - size)//2)
    img = img[y0:y0+size, x0:x0+size]
    return img

def expand_box(box, scale, W, H):
    x1,y1,x2,y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1)*scale, (y2-y1)*scale
    nx1, ny1 = max(0, int(cx-bw/2)), max(0, int(cy-bh/2))
    nx2, ny2 = min(W, int(cx+bw/2)), min(H, int(cy+bh/2))
    return nx1, ny1, nx2, ny2

def make_roi_frame_bgr(frame, dets, scale=1.35, roi_min_ratio=0.05, bg_mode="gray"):
    H, W = frame.shape[:2]
    if not dets: return None
    mask = np.zeros((H,W), np.uint8)
    for (x1,y1,x2,y2, *_ ) in dets:
        ex1,ey1,ex2,ey2 = expand_box((x1,y1,x2,y2), scale, W, H)
        mask[ey1:ey2, ex1:ex2] = 255
    if mask.mean()/255.0 < roi_min_ratio:
        return None
    if bg_mode == "blur":
        bg = cv2.GaussianBlur(frame, (0,0), sigmaX=9, sigmaY=9)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    m3 = cv2.merge([mask,mask,mask])
    return np.where(m3==255, frame, bg)

def yolo_person_boxes(yolo, frame, conf, min_area_ratio, aspect_min, aspect_max):
    r = yolo.predict(source=frame, verbose=False, conf=conf, imgsz=640)[0]
    names = getattr(r, "names", {0: "person"})
    person_ids = {i for i, n in names.items() if str(n).lower() == "person"}
    H, W = frame.shape[:2]
    min_area = min_area_ratio * (H * W)
    out = []
    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(),
                       r.boxes.cls.cpu().numpy(),
                       r.boxes.conf.cpu().numpy()):
        if (person_ids and int(c) not in person_ids) and int(c) != 0:
            continue
        x1, y1, x2, y2 = map(float, b)
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        area = w * h
        ar = w / h
        if area < min_area: continue
        if not (aspect_min <= ar <= aspect_max): continue
        out.append((x1, y1, x2, y2, float(s), int(c)))
    return out

# ================== Main ==================
def run_video(src=0):
    # load checkpoint
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes", ["assault","swoon","trespass","vandalism"])
    t_fast  = int(ckpt.get("t_fast", 32))
    alpha   = int(ckpt.get("alpha", 4))
    size    = int(ckpt.get("size", 224))
    t_slow  = max(1, t_fast // alpha)

    model = slowfast_r50(pretrained=False)
    model.blocks[-1].proj = torch.nn.Linear(model.blocks[-1].proj.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    yolo = YOLO(YOLO_WEIGHTS)

    # reader
    try:
        import imageio.v2 as iio
        HAVE_IIO=True
    except Exception:
        HAVE_IIO=False

    def open_video_any(src):
        if isinstance(src, str):
            src = os.path.normpath(src)
        cap = cv2.VideoCapture(src)
        if cap.isOpened(): return ("cv2", cap)
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if cap.isOpened(): return ("cv2", cap)
        if HAVE_IIO and isinstance(src, str):
            try:
                rdr = iio.get_reader(src); _=rdr.get_next_data(); rdr.close(); rdr=iio.get_reader(src)
                return ("imageio", rdr)
            except Exception: pass
        return (None, None)

    def read_frames(kind, handle):
        if kind=="cv2":
            cap=handle
            while True:
                ok,f=cap.read()
                if not ok: break
                yield f
            cap.release()
        elif kind=="imageio":
            rdr=handle
            for f in rdr:
                yield f[:,:,::-1].copy()
            rdr.close()
        else:
            raise RuntimeError("no reader")

    os.makedirs(OUT_DIR, exist_ok=True)
    kind, handle = open_video_any(src)
    if kind is None: raise AssertionError(f"Cannot open {src}")
    stem = "webcam" if isinstance(src,int) else os.path.splitext(os.path.basename(str(src)))[0]
    out_mp4 = os.path.join(OUT_DIR, f"{stem}_slowfast_roi_gate_pred.mp4")
    out_csv = os.path.join(OUT_DIR, f"{stem}_slowfast_roi_gate_pred.csv")

    # fps
    fps_guess = 25.0
    if kind == "cv2":
        fps = float(handle.get(cv2.CAP_PROP_FPS) or fps_guess)
    else:
        fps = fps_guess

    writer=None
    buf = deque(maxlen=t_fast)
    rows=[]; no_person_streak=0
    frame_idx=0

    print(f"[INFO] Classes={classes}  t_fast={t_fast} alpha={alpha} size={size}")
    print(f"[INFO] Output: {out_mp4} / {out_csv}")

    for frame in read_frames(kind, handle):
        frame_idx += 1
        if SAVE_VIDEO and writer is None:
            h,w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))

        # detections
        dets = yolo_person_boxes(yolo, frame, PERSON_CONF, MIN_AREA_RATIO, ASPECT_MIN, ASPECT_MAX)
        roi = make_roi_frame_bgr(frame, dets, scale=ROI_SCALE, roi_min_ratio=ROI_MIN_AREA_RATIO, bg_mode=BG_MODE)

        if roi is None:
            no_person_streak += 1
            if no_person_streak >= NO_PERSON_CLEAR:
                buf.clear()
            vis = frame.copy()
            cv2.putText(vis, "no_person", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
            for (x1,y1,x2,y2, *_ ) in dets:
                cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            if SAVE_VIDEO and writer is not None: writer.write(vis)
            if SAVE_CSV: rows.append({"frame": frame_idx, "pred":"no_person", "conf":""})
            cv2.imshow("demo", vis); 
            if cv2.waitKey(1)&0xFF==27: break
            continue
        else:
            no_person_streak=0

        rgb = center_crop_224_rgb(roi, size)  # RGB
        buf.append(rgb)

        pred_label=""; pred_conf=""
        if len(buf) == t_fast:
            # build pathways
            fast = torch.from_numpy(np.stack(list(buf),0)).permute(3,0,1,2).unsqueeze(0).float()/255.0  # (1,C,Tf,H,W)
            slow = fast[:, :, ::alpha, :, :]
            fast = (fast - MEAN) / STD
            slow = (slow - MEAN) / STD

            with torch.no_grad():
                logits = model([slow.to(device), fast.to(device)])
                p = torch.softmax(logits, dim=1)[0].cpu().numpy()
            k = int(p.argmax()); conf=float(p[k])
            pred_label = classes[k]; pred_conf=f"{conf:.4f}"

        vis = frame.copy()
        title = f"{pred_label} {pred_conf}" if pred_label else "…"
        cv2.putText(vis, title, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
        for (x1,y1,x2,y2, *_ ) in dets:
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

        if SAVE_VIDEO and writer is not None: writer.write(vis)
        if SAVE_CSV: rows.append({"frame": frame_idx, "pred": pred_label, "conf": pred_conf})

        cv2.imshow("demo", vis)
        if cv2.waitKey(1)&0xFF==27: break

    if writer is not None: writer.release()
    if SAVE_CSV and rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame","pred","conf"])
            w.writeheader(); w.writerows(rows)
    cv2.destroyAllWindows()
    print("[DONE] Saved to", OUT_DIR)

if __name__=="__main__":
    src = 0
    if len(sys.argv)>1:
        arg = sys.argv[1]
        try: src = int(arg)
        except ValueError: src = arg
    run_video(src)
