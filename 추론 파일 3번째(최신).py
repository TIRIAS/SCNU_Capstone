
import os, sys, csv, cv2, numpy as np, torch, warnings, math
from collections import deque
from pytorchvideo.models.hub import slowfast_r50
from ultralytics import YOLO

# ================== Config ==================
CKPT = r"D:\\large_data\\checkpoints\\slowfast_r50_4cls_balanced.pt"
YOLO_WEIGHTS = r"C:\\Users\\user\\runs\\detect\\train18\\weights\\best.pt"
OUT_DIR = r"D:\\large_data\\inference_out"
SAVE_VIDEO = True
SAVE_CSV = True

# --- (임시) 비활성화할 클래스들 ---
DISABLE_CLASSES = {"vandalism"}   # 3클래스 ckpt면 무시됨

# --- Detection / ROI gating ---
PERSON_CONF = 0.60
PERSON_CONF_STRICT = 0.72
MIN_AREA_RATIO = 0.0035
ASPECT_MIN, ASPECT_MAX = 0.30, 0.70
NO_PERSON_CLEAR = 8
ROI_SCALE = 1.45
ROI_MIN_AREA_RATIO = 0.02
BG_MODE = "gray"  # or "blur"

# --- 정지 물체 억제 + 정지 인원 유지(OR 게이트) ---
BOX_MOTION_THRESH = 3.0
BOX_MOTION_ERODE = 3
TRACK_IOU_KEEP = 0.45
TRACK_PERSIST_FR = 15
TRACK_SIZE_GUARD = 0.40

# --- SlowFast sampling & norm ---
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")
MEAN = torch.tensor([0.45,0.45,0.45]).view(1,3,1,1,1)
STD  = torch.tensor([0.225,0.225,0.225]).view(1,3,1,1,1)

# --- Temporal Stabilization ---
LOGIT_EMA = 0.70
MARGIN_MIN = 0.10
SWITCH_DELTA = 0.12
SWITCH_CONSEC = 5
MIN_HOLD = {"assault": 48, "swoon": 90, "trespass": 40}
MIN_SHOW_CONF = 0.30

# ====== Rule pack: assault / trespass ======
CLASS_SCALE = {"assault": 1.10, "trespass": 1.00, "swoon": 1.00}

# Assault
ASSAULT_MIN_PEOPLE   = 2
ASSAULT_NEAR_THRESH  = 0.15
ASSAULT_MOTION_GATE  = 7.0
ASSAULT_BOOST        = 1.50
ASSAULT_DAMP_SINGLE  = 0.60
KICK_RATIO_THRESH    = 1.35
KICK_MOTION_THRESH   = 8.0
KICK_BOOST           = 1.25
SWING_EDGE_THRESH    = 12.0
SWING_NEAR_THRESH    = 0.18
SWING_BOOST          = 1.20
QUIET_MOTION_THRESH  = 5.0
WEAK_KICK_RATIO      = 1.10
WEAK_SWING           = 8.0
SOLO_SUPPRESS_ASSAULT= 0.25
ASSAULT_CONTRA_FRAMES = 10
ASSAULT_SUPPRESS_TRESPASS = 0.80

# Trespass (no-zone): relax thresholds
EDGE_MARGIN_RATIO       = 0.08
MIN_OUTSIDE_FOR_ENTRY_FR= 8       # 12 → 8 (가장자리 체류 완화)
CENTRAL_MARGIN_RATIO    = 0.22    # 0.25 → 0.22 (중앙 영역 살짝 확대)
TRESPASS_STAY_FR        = 12      # 14 → 12 (중앙 체류 프레임 완화)
TRESPASS_BOOST          = 1.35
TRESPASS_DAMP_WANDER    = 0.80
MIN_INWARD_SPEED_NORM   = 0.0025  # 0.004 → 0.0025 (안쪽 속도 완화)
MIN_INWARD_DEPTH_RATIO  = 0.12    # 0.20 → 0.12 (안쪽 거리 완화)
ENTRY_TIMEOUT_FR        = 75

# Pre-entry loiter → entry boost
LOITER_WINDOW_FR        = 45
LOITER_EDGE_BAND        = 0.12
LOITER_RADIUS_NORM      = 0.04
LOITER_ENTRY_BOOST      = 1.25

# Fence jump heuristic
JUMP_WIN_FR             = 16
JUMP_VY_SPIKE           = 0.018   # 살짝 완화
JUMP_TOTAL_DY           = 0.08
JUMP_ENTRY_BOOST        = 1.30
JUMP_STAY_RELAX         = 10

# NEW: "새로 등장" 진입(문 열고 들어오기 등)
NEW_APPEAR_WINDOW_FR    = 30      # 최근 N프레임 내 인원수 증가
NEW_APPEAR_CEN_REQ      = 4       # 중앙 체류 4프레임이면 진입 인정
NEW_APPEAR_ENTRY_BOOST  = 1.20

# --- Fallbacks ---
T_FAST_FALLBACK = 32
ALPHA_FALLBACK  = 4
SIZE_FALLBACK   = 224

# ================== Utils ==================
def center_crop_rgb(bgr, size):
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

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / max(1.0, area_a + area_b - inter)

def make_roi_frame(frame, dets, scale=1.45, roi_min_ratio=0.02, bg_mode="gray"):
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

# ================== Main ==================
def run_video(src=0):
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes", ["assault","swoon","trespass","vandalism"])
    t_fast  = int(ckpt.get("t_fast", T_FAST_FALLBACK))
    alpha   = int(ckpt.get("alpha", ALPHA_FALLBACK))
    size    = int(ckpt.get("size", SIZE_FALLBACK))

    model = slowfast_r50(pretrained=False)
    model.blocks[-1].proj = torch.nn.Linear(model.blocks[-1].proj.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    yolo = YOLO(YOLO_WEIGHTS)

    try:
        import imageio.v2 as iio; HAVE_IIO=True
    except Exception:
        HAVE_IIO=False

    def open_video_any(src):
        if isinstance(src, str): src = os.path.normpath(src)
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
            for f in rdr: yield f[:,:,::-1].copy()
            rdr.close()
        else: raise RuntimeError("no reader")

    os.makedirs(OUT_DIR, exist_ok=True)
    kind, handle = open_video_any(src)
    if kind is None: raise AssertionError(f"Cannot open {src}")
    stem = "webcam" if isinstance(src,int) else os.path.splitext(os.path.basename(str(src)))[0]
    out_mp4 = os.path.join(OUT_DIR, f"{stem}_sf_rules_v35_nozone.mp4")
    out_csv = os.path.join(OUT_DIR, f"{stem}_sf_rules_v35_nozone.csv")

    fps_guess = 25.0
    if kind == "cv2": fps = float(handle.get(cv2.CAP_PROP_FPS) or fps_guess)
    else: fps = fps_guess

    writer=None
    buf = deque(maxlen=t_fast)
    rows=[]; no_person_streak=0; frame_idx=0

    p_ema = None
    current_label = ""
    hold_count = 0
    switch_count = 0

    # Rule feature states
    last_gray_rule = None
    last_edge_rule = None
    last_mean_center = None
    prev_gray_full  = None

    # Simple short-term "tracks": list of (bbox, life, last_seen_frame)
    tracks = []  # each: dict(bbox=[x1,y1,x2,y2], life=int, last=frame_idx)

    # no-zone trespass states
    outside_streak = 0
    entry_armed = False
    armed_edge = None
    armed_center = None
    armed_frames = 0
    entry_happened = False
    central_streak = 0

    # loiter and jump buffers
    edge_centers = deque(maxlen=LOITER_WINDOW_FR)
    vy_buffer = deque(maxlen=JUMP_WIN_FR)

    # NEW: "new appearance" entry states
    last_person_count = 0
    spawn_armed = False
    spawn_timer = 0

    # assault contradiction
    assault_contra = 0
    solo_suppressed = False

    print(f"[INFO] Classes={classes}  disabled={DISABLE_CLASSES}")
    print(f"[INFO] Output: {out_mp4} / {out_csv}")

    for frame in read_frames(kind, handle):
        frame_idx += 1
        if SAVE_VIDEO and writer is None:
            h,w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # YOLO → raw candidates
        r = yolo.predict(source=frame, verbose=False, conf=PERSON_CONF, imgsz=640)[0]
        names = getattr(r, "names", {0: "person"})
        person_ids = {i for i, n in names.items() if str(n).lower() == "person"}
        H, W = frame.shape[:2]
        min_area = MIN_AREA_RATIO * (H * W)

        # update simple tracks decay
        tracks = [t for t in tracks if frame_idx - t["last"] <= TRACK_PERSIST_FR]
        for t in tracks: t["life"] = min(TRACK_PERSIST_FR, t["life"]+1)

        dets=[]
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(),
                           r.boxes.cls.cpu().numpy(),
                           r.boxes.conf.cpu().numpy()):
            if (person_ids and int(c) not in person_ids) and int(c) != 0:
                continue
            x1, y1, x2, y2 = map(int, b)
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            area = bw * bh
            ar = bw / bh
            if area < min_area: 
                continue
            if not (ASPECT_MIN <= ar <= ASPECT_MAX): 
                continue

            # per-box motion
            mval = 0.0
            if prev_gray_full is not None:
                ex1,ey1,ex2,ey2 = max(0,x1),max(0,y1),min(W,x2),min(H,y2)
                if ex2>ex1 and ey2>ey1:
                    patch_prev = prev_gray_full[ey1:ey2, ex1:ex2]
                    patch_curr = frame_gray[ey1:ey2, ex1:ex2]
                    if patch_prev.shape == patch_curr.shape and patch_prev.size > 0:
                        diff = cv2.absdiff(patch_curr, patch_prev)
                        if BOX_MOTION_ERODE>0:
                            k=BOX_MOTION_ERODE
                            diff = cv2.erode(diff, np.ones((k,k), np.uint8), 1)
                        mval = float(diff.mean())

            is_moving = mval >= BOX_MOTION_THRESH
            is_human_shape = ASPECT_MIN <= ar <= ASPECT_MAX
            is_strict = float(s) >= PERSON_CONF_STRICT

            # short-term track match
            keep_by_track = False
            b_now = [float(x1), float(y1), float(x2), float(y2)]
            if tracks and (bw/W <= TRACK_SIZE_GUARD and bh/H <= TRACK_SIZE_GUARD):
                best=None; best_iou=0.0
                for t in tracks:
                    i = iou(b_now, t["bbox"])
                    if i>best_iou: best_iou=i; best=t
                if best is not None and best_iou >= TRACK_IOU_KEEP:
                    keep_by_track = True
                    best["bbox"] = b_now; best["last"] = frame_idx

            if not keep_by_track:
                if is_moving or (is_human_shape and is_strict):
                    tracks.append({"bbox": b_now, "life":1, "last":frame_idx})

            keep = is_moving or (is_human_shape and is_strict) or keep_by_track
            if keep:
                dets.append((float(x1), float(y1), float(x2), float(y2), float(s), int(c)))

        prev_gray_full = frame_gray.copy()

        # --- NEW APPEAR: count-based arming ---
        person_count = len(dets)
        if person_count > last_person_count:
            spawn_armed = True
            spawn_timer = 0
        last_person_count = person_count
        if spawn_armed:
            spawn_timer += 1
            if spawn_timer > NEW_APPEAR_WINDOW_FR:
                spawn_armed = False

        roi = make_roi_frame(frame, dets, scale=ROI_SCALE, roi_min_ratio=ROI_MIN_AREA_RATIO, bg_mode=BG_MODE)

        if roi is None:
            no_person_streak += 1
            if no_person_streak >= NO_PERSON_CLEAR:
                buf.clear(); p_ema=None; current_label=""; hold_count=0; switch_count=0
                last_gray_rule=None; last_edge_rule=None; last_mean_center=None
                outside_streak=0; entry_armed=False; entry_happened=False; central_streak=0
                armed_edge=None; armed_center=None; armed_frames=0
                edge_centers.clear(); vy_buffer.clear()
                spawn_armed=False; spawn_timer=0; last_person_count=0
                assault_contra=0; solo_suppressed=False
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

        # ---------- 규칙 피처 ----------
        H, W = frame.shape[:2]
        diag = (H**2 + W**2) ** 0.5

        centers = []
        for (x1,y1,x2,y2,*_) in dets:
            cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
            centers.append((cx,cy))

        mean_center = None
        if centers:
            mean_center = (float(np.mean([c[0] for c in centers])), float(np.mean([c[1] for c in centers])))

        v = (0.0,0.0)
        vy_norm = 0.0
        if mean_center is not None and last_mean_center is not None:
            dx = mean_center[0] - last_mean_center[0]
            dy = mean_center[1] - last_mean_center[1]
            v = (dx, dy)
            vy_norm = dy / max(1.0, H)
        last_mean_center = mean_center

        # 최소 쌍 거리(정규화)
        min_pair_dist = 1.0
        if len(centers) >= 2:
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    d = (dx*dx + dy*dy) ** 0.5 / max(1.0, diag)
                    if d < min_pair_dist: min_pair_dist = d

        # ROI 차영상
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        diff = None
        motion_val = 0.0
        if last_gray_rule is not None and last_gray_rule.shape == roi_gray.shape:
            diff = cv2.absdiff(roi_gray, last_gray_rule)
            motion_val = float(np.mean(diff))
        last_gray_rule = roi_gray

        # 하체/상체 모션 분할
        kick_ratio = 0.0; lower_val = 0.0; upper_val = 0.0
        if diff is not None:
            h2 = diff.shape[0]//2
            upper_val = float(np.mean(diff[:h2,:]))
            lower_val = float(np.mean(diff[h2:,:]))
            kick_ratio = lower_val / max(1e-5, upper_val)

        # 에지-모션(상체 근사)
        swing_val = 0.0
        edges = cv2.Canny(roi_gray, 50, 150)
        if last_edge_rule is not None and last_edge_rule.shape == edges.shape:
            ediff = cv2.absdiff(edges, last_edge_rule)
            h2 = edges.shape[0]//2
            swing_val = float(np.mean(ediff[:h2,:]))
        last_edge_rule = edges

        # ----- NO-ZONE trespass: edge-entry + central dwell + inward gating -----
        edge_margin_w = EDGE_MARGIN_RATIO * W
        edge_margin_h = EDGE_MARGIN_RATIO * H
        near_edge = False
        nearest_edge = None
        if mean_center is not None:
            x, y = mean_center
            dists = {"left": x, "right": W - x, "top": y, "bottom": H - y}
            nearest_edge = min(dists, key=dists.get)
            if (x < edge_margin_w) or (x > W - edge_margin_w) or (y < edge_margin_h) or (y > H - edge_margin_h):
                near_edge = True

        # loiter ring buffer (edge band)
        if mean_center is not None:
            x, y = mean_center
            if (x < LOITER_EDGE_BAND*W) or (x > (1-LOITER_EDGE_BAND)*W) or (y < LOITER_EDGE_BAND*H) or (y > (1-LOITER_EDGE_BAND)*H):
                edge_centers.append((x,y))
            else:
                edge_centers.clear()

        # 장전(arming)
        if near_edge:
            outside_streak += 1
            if outside_streak >= MIN_OUTSIDE_FOR_ENTRY_FR:
                entry_armed = True
                armed_edge = nearest_edge
                armed_center = mean_center
                armed_frames = 0
        else:
            outside_streak = 0

        # 장전 유지/타임아웃
        if entry_armed:
            armed_frames += 1
            if armed_frames > ENTRY_TIMEOUT_FR:
                entry_armed = False
                armed_edge = None
                armed_center = None
                armed_frames = 0
                entry_happened = False

        # 진입 판단
        if entry_armed and mean_center is not None and armed_center is not None and armed_edge is not None:
            normals = {"left": (1,0), "right": (-1,0), "top": (0,1), "bottom": (0,-1)}
            nx, ny = normals[armed_edge]
            inward_speed = ((v[0]*nx + v[1]*ny) / max(1.0, diag))
            depth_pix = (mean_center[0]-armed_center[0])*nx + (mean_center[1]-armed_center[1])*ny
            denom = (W if armed_edge in ("left","right") else H)
            depth_ratio = (depth_pix / denom) if depth_pix>0 else 0.0
            if inward_speed >= MIN_INWARD_SPEED_NORM and depth_ratio >= MIN_INWARD_DEPTH_RATIO:
                entry_happened = True
                entry_armed = False
                armed_edge = None
                armed_center = None
                armed_frames = 0

        # 중앙 영역 체류
        central = False
        if mean_center is not None:
            cx, cy = mean_center
            cmx = CENTRAL_MARGIN_RATIO * W
            cmy = CENTRAL_MARGIN_RATIO * H
            if (cmx <= cx <= W - cmx) and (cmy <= cy <= H - cmy):
                central = True
        if central and entry_happened:
            central_streak += 1
        else:
            central_streak = max(0, central_streak - 1)

        # ----- Pre-entry loiter boost -----
        loiter_boost = 1.0
        if len(edge_centers) >= LOITER_WINDOW_FR:
            xs = np.array([p[0] for p in edge_centers]); ys = np.array([p[1] for p in edge_centers])
            cx_, cy_ = xs.mean(), ys.mean()
            rad = np.mean(np.sqrt((xs-cx_)**2 + (ys-cy_)**2)) / max(1.0, math.sqrt(W*W+H*H))
            if rad >= LOITER_RADIUS_NORM:
                loiter_boost = LOITER_ENTRY_BOOST

        # ----- Fence jump heuristic -----
        jump_boost = 1.0
        if mean_center is not None:
            vy_buffer.append(vy_norm)
            if len(vy_buffer) >= 4:
                total_dy = abs(sum(vy_buffer))
                max_vy = max(abs(x) for x in vy_buffer)
                if max_vy >= JUMP_VY_SPIKE and total_dy >= JUMP_TOTAL_DY:
                    jump_boost = JUMP_ENTRY_BOOST

        # ----- NEW APPEAR → entry -----
        spawn_boost = 1.0
        if spawn_armed and central_streak >= NEW_APPEAR_CEN_REQ:
            entry_happened = True
            spawn_boost = NEW_APPEAR_ENTRY_BOOST
            spawn_armed = False

        # ---------- 분류 입력 ----------
        roi_rgb = center_crop_rgb(roi, size)  # RGB
        buf.append(roi_rgb)

        pred_label=""; pred_conf=""
        solo_suppressed = False
        if len(buf) == t_fast:
            fast = torch.from_numpy(np.stack(list(buf),0)).permute(3,0,1,2).unsqueeze(0).float()/255.0
            slow = fast[:, :, ::alpha, :, :]
            fast = (fast - MEAN) / STD
            slow = (slow - MEAN) / STD

            with torch.no_grad():
                logits = model([slow.to(device), fast.to(device)])
                p = torch.softmax(logits, dim=1)[0].cpu().numpy()

            # 비활성화
            mask = np.ones_like(p, dtype=np.float32)
            for i, cls in enumerate(classes):
                if cls in DISABLE_CLASSES: mask[i] = 0.0
            p = p * mask

            # 기본 스케일
            for i, cls in enumerate(classes):
                p[i] *= CLASS_SCALE.get(cls, 1.0)

            # Assault 규칙
            assault_like = False
            if "assault" in classes:
                ia = classes.index("assault")
                base_cond = (len(centers) >= ASSAULT_MIN_PEOPLE and
                             min_pair_dist <= ASSAULT_NEAR_THRESH and
                             motion_val >= ASSAULT_MOTION_GATE)
                kick_cond = (kick_ratio >= KICK_RATIO_THRESH and lower_val >= KICK_MOTION_THRESH)
                swing_cond = (swing_val >= SWING_EDGE_THRESH and min_pair_dist <= SWING_NEAR_THRESH)
                if base_cond: p[ia] *= ASSAULT_BOOST; assault_like=True
                if kick_cond: p[ia] *= KICK_BOOST; assault_like=True
                if swing_cond: p[ia] *= SWING_BOOST; assault_like=True
                if (len(centers) < 2 and motion_val < QUIET_MOTION_THRESH and
                    kick_ratio < WEAK_KICK_RATIO and swing_val < WEAK_SWING):
                    p[ia] *= SOLO_SUPPRESS_ASSAULT; solo_suppressed=True

            # Trespass 규칙 (no-zone + inward + loiter/jump/spawn 부스트)
            if "trespass" in classes:
                it = classes.index("trespass")
                if entry_happened and central_streak >= TRESPASS_STAY_FR:
                    p[it] *= TRESPASS_BOOST * loiter_boost * jump_boost * spawn_boost
                else:
                    p[it] *= TRESPASS_DAMP_WANDER
                if assault_like:
                    p[it] *= ASSAULT_SUPPRESS_TRESPASS

            s = float(p.sum())
            if s > 1e-8: p /= s
            else: p[:] = 0.0

            if p_ema is None:
                p_ema = p.copy()
            else:
                p_ema = LOGIT_EMA * p_ema + (1.0 - LOGIT_EMA) * p

            # assault 모순 해제
            if "assault" in classes:
                if (len(centers) < 2 or min_pair_dist > 0.30) and motion_val < QUIET_MOTION_THRESH:
                    assault_contra += 1
                else:
                    assault_contra = 0
                if current_label == "assault" and assault_contra >= ASSAULT_CONTRA_FRAMES:
                    current_label = ""; hold_count = 0; switch_count = 0; assault_contra = 0

            # 히스테리시스
            order = np.argsort(-p_ema)
            k1 = int(order[0])
            k2 = int(order[1]) if len(order) > 1 else k1
            top1, top2 = float(p_ema[k1]), float(p_ema[k2])
            label1 = classes[k1]

            if current_label == "":
                if top1 >= MIN_SHOW_CONF and (top1 - top2) >= MARGIN_MIN:
                    current_label = label1; hold_count = 1; switch_count = 0
            else:
                if label1 == current_label:
                    hold_count += 1; switch_count = 0
                else:
                    curr_idx = classes.index(current_label)
                    cond_delta = (top1 - float(p_ema[curr_idx])) >= SWITCH_DELTA
                    cond_margin = (top1 - top2) >= MARGIN_MIN
                    cond_hold = hold_count >= MIN_HOLD.get(current_label, 16)
                    if cond_delta and cond_margin and cond_hold:
                        switch_count += 1
                        if switch_count >= SWITCH_CONSEC:
                            current_label = label1; hold_count = 1; switch_count = 0
                    else:
                        switch_count = 0

            if current_label and top1 >= MIN_SHOW_CONF:
                pred_label = current_label; pred_conf = f"{top1:.2f}"

        # ---------- 시각화 ----------
        vis = frame.copy()
        title = f"{pred_label} {pred_conf}" if pred_label else "???"
        for (x1,y1,x2,y2, *_ ) in dets:
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

        dbg = f"near={min_pair_dist:.2f} mot={motion_val:.1f} cen={central_streak} out={outside_streak} ent={int(entry_happened)} loit={len(edge_centers)} spawn={int(spawn_armed)}"
        cv2.putText(vis, title, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
        cv2.putText(vis, dbg, (12,52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

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
