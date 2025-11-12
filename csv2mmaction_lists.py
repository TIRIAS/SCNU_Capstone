# D:\i3d_tools\csv2mmaction_lists.py
import csv, sys
from pathlib import Path

root = Path(r"D:\i3d_abn50")  # build_i3d_from_nia.py의 --dst
for split in ["train","val"]:
    in_csv = root / f"{split}.csv"
    out_txt = root / f"{split}_list.txt"
    with open(in_csv, newline="", encoding="utf-8") as f, open(out_txt, "w", encoding="utf-8") as w:
        r = csv.DictReader(f)
        for row in r:
            rel = row["relpath"].replace("\\","/")
            label = int(row["label"])
            w.write(f"{rel} {label}\n")
    print(f"[OK] {out_txt}")
print("[DONE]")
