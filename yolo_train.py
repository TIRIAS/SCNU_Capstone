from train import train
from utils.torch_utils import select_device

def main():
    train(
        data='D:/CCTV/CCTV/frames/data.yaml',  # data.yaml 경로
        cfg='models/yolo.yaml',                # YOLO 모델 구조 설정
        weights='',                            # 사전학습 가중치 경로 (없으면 ''로 시작)
        epochs=50,                             # 학습 에폭 수
        batch_size=16,                         # 배치 사이즈
        imgsz=640,                             # 이미지 입력 크기
        device=select_device('0')              # GPU: '0', CPU: 'cpu'
    )

if __name__ == '__main__':
    main()
