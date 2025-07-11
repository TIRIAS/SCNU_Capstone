import os
import shutil
import random
from glob import glob

# 원본 경로
BASE_DIR = r'D:\CCTV\CCTV\frames'

# 출력 경로
IMAGE_TRAIN = os.path.join(BASE_DIR, 'images', 'train')
IMAGE_VAL   = os.path.join(BASE_DIR, 'images', 'val')
LABEL_TRAIN = os.path.join(BASE_DIR, 'labels', 'train')
LABEL_VAL   = os.path.join(BASE_DIR, 'labels', 'val')

# 폴더 생성
for d in [IMAGE_TRAIN, IMAGE_VAL, LABEL_TRAIN, LABEL_VAL]:
    os.makedirs(d, exist_ok=True)

# 클래스 폴더 목록
class_folders = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR)
                 if os.path.isdir(os.path.join(BASE_DIR, d)) and '.' in d]

# 비율
train_ratio = 0.8

for class_folder in class_folders:
    images = sorted(glob(os.path.join(class_folder, '*.jpg')))  # 또는 .png
    random.shuffle(images)

    split = int(len(images) * train_ratio)
    train_images = images[:split]
    val_images = images[split:]

    for img_list, img_dst, lbl_dst in [
        (train_images, IMAGE_TRAIN, LABEL_TRAIN),
        (val_images, IMAGE_VAL, LABEL_VAL)
    ]:
        for img_path in img_list:
            name = os.path.basename(img_path)
            label_path = img_path.replace('.jpg', '.txt')  # 확장자에 따라 수정

            # 복사
            shutil.copy(img_path, os.path.join(img_dst, name))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(lbl_dst, os.path.basename(label_path)))
            else:
                print(f'❗ 라벨 없음: {label_path}')
