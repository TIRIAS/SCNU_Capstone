# -*- coding: utf-8 -*-
import os, sys, re, json, math, random, shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

# --------- 고정 매핑(요청 사양) ----------
CLASS2ID = {
    "assault": 0,
    "swoon": 1,
    "trespass": 2,
    "vandalism": 3,
}
ID2CLASS = {v:k for k,v in CLASS2ID.items()}

# --------- 유틸 ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_hhmmss_to_frames(s: str, fps: float) -> int:
    # "HH:MM:SS.s" 혹은 "00:04:3.3" 처럼 M 또는 S 앞의 0이 빠진 포맷도 허용
    parts = s.strip().split(':')
    if len(parts) != 3:
        raise ValueError(f"time format not HH:MM:SS.s: {s}")
    hh = float(parts[0]); mm = float(parts[1]); ss = float(parts[2])
    total_sec = hh*3600 + mm*60 + ss
    return int(round(total_sec * fps))

def parse_xml(xml_path: Path):
    """
    반환: dict(fps, frames, eventname, action_ranges=[(s,e),...])
    action_ranges는 프레임 인덱스 [start, end) 관례로 처리
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def find_text(path, default=None):
        node = root.find(path)
        return node.text.strip() if node is not None and node.text else default

    fps_txt = find_text('header/fps', '30')
    frames_txt = find_text('header/frames', None)
    fps = float(fps_txt)
    frames = int(frames_txt) if frames_txt else None

    eventname = find_text('event/eventname', None)

    # action 프레임 범위들 수집
    action_ranges = []
    for act in root.findall('object/action'):
        s1 = act.find('frame/start')
        e1 = act.find('frame/end')
        # 일부 XML은 동일 <action>에 frame 태그가 두 번 반복됨 → 모두 수집
        frames_tags = act.findall('frame')
        if frames_tags and len(frames_tags) >= 1:
            for ft in frames_tags:
                s = int(ft.find('start').text)
                e = int(ft.find('end').text)
                if e > s:
                    action_ranges.append((s, e))
        elif s1 is not None and e1 is not None:
            s = int(s1.text); e = int(e1.text)
            if e > s:
                action_ranges.append((s, e))

    # action이 없으면 event(start+duration)로 대체
    if not action_ranges:
        st_txt = find_text('event/starttime', None)
        du_txt = find_text('event/duration', None)
        if st_txt and du_txt:
            st = parse_hhmmss_to_frames(st_txt, fps)
            # duration HH:MM:SS.s → 초
            parts = du_txt.split(':')
            hh = float(parts[0]); mm = float(parts[1]); ss = float(parts[2])
            dur_frames = int(round((hh*3600 + mm*60 + ss) * fps))
            action_ranges.append((st, st + dur_frames))

    # 겹치는 구간 병합
    action_ranges.sort()
    merged = []
    for s,e in action_ranges:
        if not merged or s > merged[-1][1]:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    action_ranges = [(s,e) for s,e in merged]

    return dict(fps=fps, frames=frames, eventname=eventname, action_ranges=action_ranges)

def frame_in_ranges(idx, ranges):
    for s,e in ranges:
        if s <= idx < e:
            return True
    return False

# --------- 의사 박스(사람) 생성용 ----------
def get_person_detector(model_name="yolo11s.pt", imgsz=1280, conf=0.45, device=0):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERROR] ultralytics가 필요합니다: pip install ultralytics")
        raise
    model = YOLO(model_name)
    # People class only
    model.overrides.update(dict(imgsz=imgsz, conf=conf, device=device, classes=[0]))
    return model

def persons_to_yolo_labels(results, W, H, max_people=8, min_rel=0.02):
    """
    results: model(frame) 반환값 중 첫 번째의 boxes 사용
    min_rel: 최소 상대크기(너비,높이) 비율
    """
    labels = []
    if not results or len(results) == 0:
        return labels
    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return labels
    # boxes.xyxy: (N,4), boxes.conf, boxes.cls
    # 크기 순 정렬(큰 사람 우선)
    b = r.boxes
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    areas = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1])
    order = areas.argsort()[::-1]
    count = 0
    for i in order:
        x1,y1,x2,y2 = xyxy[i]
        w = max(1.0, x2-x1); h = max(1.0, y2-y1)
        if w/W < min_rel or h/H < min_rel:
            continue
        # YOLO xywh norm
        cx = (x1+x2)/2.0 / W
        cy = (y1+y2)/2.0 / H
        nw = w / W
        nh = h / H
        labels.append((cx, cy, nw, nh))
        count += 1
        if count >= max_people:
            break
    return labels

# --------- 메인 빌드 ----------
def build(
    src_root, dst_root,
    split=(0.7,0.2,0.1),
    stride_pos=5, stride_neg=6,
    pad_frames=10,
    neg_per_pos=1.5,
    seed=42,
    person_model="yolo11s.pt",
    person_conf=0.45,
    person_imgsz=1280,
    device=0,
    max_people=8
):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    random.seed(seed)

    # 출력 구조
    for sub in ["images/train","images/val","images/test","labels/train","labels/val","labels/test"]:
        ensure_dir(dst_root/sub)

    # 비디오 목록 수집(비디오 단위 split)
    video_items = []  # (class_name, video_path, xml_path)
    for cls in CLASS2ID.keys():
        cdir = src_root/cls
        if not cdir.is_dir():
            print(f"[WARN] 클래스 폴더 없음: {cdir}")
            continue
        for v in sorted(cdir.glob("*.mp4")):
            stem = v.stem
            # xml 후보: 동일 스템, 혹은 *_old.xml 등
            xml = None
            cand = list(cdir.glob(stem + "*.xml"))
            if cand:
                xml = cand[0]
            if xml is None:
                print(f"[WARN] XML 없음(건너뜀): {v}")
                continue
            video_items.append((cls, v, xml))

    # split
    random.shuffle(video_items)
    n = len(video_items)
    n_train = int(n*split[0]); n_val = int(n*split[1])
    splits = {"train": video_items[:n_train],
              "val": video_items[n_train:n_train+n_val],
              "test": video_items[n_train+n_val:]}

    # 사람 검출기
    person_det = get_person_detector(person_model, imgsz=person_imgsz, conf=person_conf, device=device)

    total_pos, total_neg = 0, 0

    for sp, items in splits.items():
        print(f"\n=== [{sp}] videos: {len(items)} ===")
        for cls, vpath, xpath in items:
            meta = parse_xml(xpath)
            fps = meta["fps"]; frames_total = meta["frames"]
            eventname = meta["eventname"]; ranges = meta["action_ranges"]

            # 폴더명-XML의 eventname 불일치 경고
            if eventname and eventname.lower() != cls.lower():
                print(f"[WARN] 폴더({cls}) vs XML({eventname}) 불일치: {vpath.name}")

            # 패딩
            pranges = []
            for s,e in ranges:
                ps = max(0, s - pad_frames)
                pe = e + pad_frames
                pranges.append((ps,pe))

            # 프레임 선택
            pos_idx = set()
            neg_idx = set()

            # 양성
            for s,e in pranges:
                for fidx in range(s, e, stride_pos):
                    pos_idx.add(fidx)

            # 음성(하드 네거티브): 이벤트 밖
            if frames_total is None:
                # 모르면 비디오 길이 읽어가며 샘플링
                # 나중에 캡을 열어 전체 길이를 세되, 여기선 일단 넉넉히 후보 생성
                pass

            # 캡 열어서 실제 루프
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                print(f"[ERROR] open fail: {vpath}")
                continue

            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frames_total is None:
                frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 음성 후보 샘플
            bg_pool = [i for i in range(0, frames_total, stride_neg) if not frame_in_ranges(i, pranges)]
            want_neg = int(len(pos_idx)*neg_per_pos)
            random.shuffle(bg_pool)
            for i in bg_pool[:want_neg]:
                neg_idx.add(i)

            target_idx = sorted(pos_idx | neg_idx)
            if not target_idx:
                cap.release()
                print(f"[WARN] 추출 프레임 없음: {vpath.name}")
                continue

            # 읽기 루프
            idx_set = set(target_idx)
            cur = -1
            # 빠른 점프용
            def grab_to(target):
                nonlocal cur
                if target <= cur:
                    return
                # OpenCV 정확 시킹: 일부 코덱은 정확도 떨어질 수 있어 느리지만 프레임마다 읽는다
                while cur < target:
                    ret = cap.grab()
                    if not ret:
                        break
                    cur += 1

            for fidx in target_idx:
                grab_to(fidx)
                ret, frame = cap.retrieve()
                if not ret:
                    break

                # 파일명
                stem = f"{vpath.stem}_{fidx:06d}"
                img_out = dst_root / f"images/{sp}/{stem}.jpg"
                lbl_out = dst_root / f"labels/{sp}/{stem}.txt"

                # 저장
                cv2.imwrite(str(img_out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if fidx in pos_idx:
                    # 사람 검출 → 박스 생성
                    results = person_det.predict(source=frame, verbose=False)
                    boxes = persons_to_yolo_labels(results, W, H, max_people=max_people)
                    if boxes:
                        with open(lbl_out, 'w', encoding='utf-8') as f:
                            for (cx,cy,nw,nh) in boxes:
                                f.write(f"{CLASS2ID[cls]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                        total_pos += 1
                    else:
                        # 사람 없으면 해당 프레임은 음성으로 취급(빈 라벨)
                        lbl_out.touch()
                        total_neg += 1
                else:
                    # 음성: 빈 라벨 파일
                    lbl_out.touch()
                    total_neg += 1

            cap.release()

    # data.yaml 작성
    data_yaml = dst_root / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            "path: " + str(dst_root).replace("\\","/") + "\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"names: {json.dumps([ID2CLASS[i] for i in range(len(ID2CLASS))])}\n"
        )
    print("\n[DONE] dataset ->", dst_root)
    print(f"[STATS] positive frames(images with boxes): {total_pos}, negatives(empty labels): {total_neg}")

if __name__ == "__main__":
    # 간단한 인자 파서(Windows cmd 편의)
    # 예) python build_large_dataset.py "D:\large_data" "D:\large_dataset"
    src = sys.argv[1] if len(sys.argv) > 1 else r"D:\large_data"
    dst = sys.argv[2] if len(sys.argv) > 2 else r"D:\large_dataset"
    # 필요 시 하이퍼파라미터를 환경변수로도 조정 가능
    build(
        src_root=src,
        dst_root=dst,
        split=(0.7,0.2,0.1),
        stride_pos=3,      # 이벤트 구간 양성 추출 간격
        stride_neg=6,      # 배경 추출 간격
        pad_frames=30,     # 이벤트 앞뒤 패딩(≈1초@30fps)
        neg_per_pos=1.0,   # 하드네거티브 비율
        seed=42,
        person_model=os.environ.get("PERSON_MODEL","yolo11s.pt"),
        person_conf=float(os.environ.get("PERSON_CONF","0.45")),
        person_imgsz=int(os.environ.get("PERSON_IMGSZ","1280")),
        device=int(os.environ.get("DEVICE","0")),
        max_people=int(os.environ.get("MAX_PEOPLE","8")),
    )
