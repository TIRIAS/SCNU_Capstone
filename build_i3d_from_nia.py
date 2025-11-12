# D:\i3d_tools\build_i3d_from_nia.py
import os, re, csv, random, math, argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

CLASS2ID = {"assault":0, "swoon":1, "trespass":2, "vandalism":3}

def parse_timecode(tc: str) -> float:
    # "HH:MM:SS.s" 혹은 "HH:MM:SS" 형태
    h, m, s = tc.strip().split(":")
    return int(h)*3600 + int(m)*60 + float(s)

def load_xml_info(xml_path: Path):
    """
    반환:
      {
        'event': eventname(str),
        'fps': fps(int),
        'frames': total_frames(int),
        'event_start_f': int,
        'event_end_f': int
      }
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # 필드
    ev_name = root.findtext("event/eventname").strip().lower()
    fps = int(float(root.findtext("header/fps")))
    total_frames = int(root.findtext("header/frames"))

    st_tc = root.findtext("event/starttime").strip()
    dur_tc = root.findtext("event/duration").strip()

    st_s = parse_timecode(st_tc)
    dur_s = parse_timecode(dur_tc)
    ed_s = st_s + dur_s

    # 프레임 환산
    st_f = max(0, int(round(st_s * fps)))
    ed_f = min(total_frames-1, int(round(ed_s * fps)))

    # 안전 장치
    if ed_f <= st_f:
        ed_f = min(total_frames-1, st_f + fps)  # 최소 1초 보장

    return {
        "event": ev_name,
        "fps": fps,
        "frames": total_frames,
        "event_start_f": st_f,
        "event_end_f": ed_f
    }

def pick_topN_pairs(class_dir: Path, N: int):
    mp4s = sorted(class_dir.glob("*.mp4"))
    pairs = []
    for m in mp4s:
        x = m.with_suffix(".xml")
        if x.exists():
            pairs.append((m, x))
        if len(pairs) >= N:
            break
    return pairs

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clamp(a, lo, hi):
    return max(lo, min(hi, a))

def cut_clip_cv2(in_mp4: Path, out_mp4: Path, start_f: int, end_f: int, fps: int):
    """OpenCV로 [start_f, end_f] 구간을 그대로 저장 (H.264)"""
    cap = cv2.VideoCapture(str(in_mp4))
    if not cap.isOpened():
        print(f"[WARN] cannot open {in_mp4}")
        return 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 코덱이 없다면 mp4v로
    out = cv2.VideoWriter(str(out_mp4), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    n_written = 0
    while True:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos > end_f:
            break
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        n_written += 1

    out.release()
    cap.release()
    return n_written

def make_splits(items, val_ratio=0.2, seed=42):
    random.Random(seed).shuffle(items)
    n = len(items)
    v = int(round(n * val_ratio))
    return items[v:], items[:v]  # train, val

def gen_non_overlap_intervals(total_f, pos_intervals, guard=0):
    # pos_intervals: [(s,e), ...]
    # guard: 이벤트 경계 전후 보호프레임
    mask = [0]*(total_f+1)
    for s,e in pos_intervals:
        s = clamp(s-guard, 0, total_f)
        e = clamp(e+guard, 0, total_f)
        for i in range(s, e+1):
            mask[i] = 1
    # 연속 0 구간을 음성 후보로 나열
    neg = []
    i = 0
    while i <= total_f:
        if mask[i] == 0:
            j = i
            while j <= total_f and mask[j] == 0:
                j += 1
            neg.append((i, j-1))
            i = j
        else:
            i += 1
    return neg

def sample_windows(intervals, clip_len, stride, max_windows=None):
    """간단 슬라이딩 윈도우 생성 (start,end 포함)"""
    wins = []
    for (s,e) in intervals:
        L = e - s + 1
        if L < clip_len:
            continue
        cur = s
        while cur + clip_len - 1 <= e:
            wins.append((cur, cur + clip_len - 1))
            cur += stride
    if max_windows is not None and len(wins) > max_windows:
        random.shuffle(wins)
        wins = wins[:max_windows]
    return wins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=r"예: D:\large_data")
    ap.add_argument("--dst", required=True, help=r"예: D:\i3d_abn50")
    ap.add_argument("--per_class", type=int, default=50)
    ap.add_argument("--clip_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--neg_per_pos", type=float, default=1.0)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    ensure_dir(dst)

    # 선택된 (mp4, xml, cls) 목록 수집
    selected = []
    for cname in CLASS2ID.keys():
        cdir = src / cname
        pairs = pick_topN_pairs(cdir, args.per_class)
        for m,x in pairs:
            selected.append((m, x, cname))
        print(f"[PICK] {cname}: {len(pairs)} files")

    # 비디오별로 클립 구간 산출
    meta = []  # (in_mp4, out_rel_path, label_id, fps, nframes_written)
    tmp_pairs = []

    for (mp4_path, xml_path, cname) in selected:
        info = load_xml_info(xml_path)
        lbl = CLASS2ID[cname]

        # 이벤트명-폴더 불일치 경고만 표시 (훈련 라벨은 폴더 기준으로 갑니다)
        ev = info["event"]
        if ev != cname:
            print(f"[WARN] event='{ev}' xml vs folder='{cname}' mismatch: {xml_path.name}")

        fps = info["fps"]
        total_f = info["frames"]
        pos_intervals = [(info["event_start_f"], info["event_end_f"])]

        # 양성 슬라이딩 윈도우
        pos_wins = sample_windows(pos_intervals, args.clip_len, args.stride)

        # 하드네거티브: 이벤트 경계 ±fps(1초) 보호 후, 나머지에서 샘플
        neg_intervals = gen_non_overlap_intervals(total_f, pos_intervals, guard=fps)
        # 음성 창을 너무 많이 만들지 않도록 양성과 1:1 비율
        max_neg = int(math.ceil(len(pos_wins) * args.neg_per_pos))
        neg_wins = sample_windows(neg_intervals, args.clip_len, args.clip_len, max_windows=max_neg)

        tmp_pairs.append((mp4_path, cname, fps, pos_wins, neg_wins))

    # train/val로 파일 단위 분할(비디오 단위로 나눔)
    train_items, val_items = make_splits(tmp_pairs, val_ratio=args.val_ratio, seed=args.seed)

    def dump_split(items, split):
        split_root = dst / split
        for (_, cname, _, _, _) in items:
            ensure_dir(split_root / cname)

        records = []
        for (mp4_path, cname, fps, pos_wins, neg_wins) in items:
            # 출력 파일명 접두
            stem = mp4_path.stem

            # 양성
            for idx, (s,e) in enumerate(pos_wins):
                out_rel = Path(split) / cname / f"{stem}_pos_{s}-{e}.mp4"
                out_abs = dst / out_rel
                nfr = cut_clip_cv2(mp4_path, out_abs, s, e, fps)
                if nfr > 0:
                    records.append([str(out_rel).replace("\\","/"), CLASS2ID[cname], fps, nfr])

            # 음성
            for idx, (s,e) in enumerate(neg_wins):
                out_rel = Path(split) / cname / f"{stem}_neg_{s}-{e}.mp4"
                out_abs = dst / out_rel
                nfr = cut_clip_cv2(mp4_path, out_abs, s, e, fps)
                if nfr > 0:
                    records.append([str(out_rel).replace("\\","/"), CLASS2ID[cname], fps, nfr])

        # CSV 저장
        csv_path = dst / f"{split}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["relpath","label","fps","num_frames"])
            w.writerows(records)
        print(f"[{split.upper()}] clips={len(records)} -> {csv_path}")

    dump_split(train_items, "train")
    dump_split(val_items,   "val")

    # 라벨 맵 저장
    with open(dst/"labels.txt","w",encoding="utf-8") as f:
        for k,v in CLASS2ID.items():
            f.write(f"{v}\t{k}\n")

    print("[DONE] I3D dataset ready at:", dst)

if __name__ == "__main__":
    main()
