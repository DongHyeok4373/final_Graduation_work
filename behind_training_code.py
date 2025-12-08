from ultralytics import YOLO
import torch

def main():
    # Pretrained pose model 불러오기
    model = YOLO(r"C:\Users\ehdgu\Desktop\2people\코딩\코딩\behind_runs\exp_behind116\weights\best.pt")

    # right_foot 전용 데이터셋 yaml
    data_yaml = r"C:\Users\ehdgu\Desktop\2people\코딩\코딩\foot_dataset_updated\behind_view\data.yaml"

    model.train(
        data=data_yaml,
        epochs=500,
        imgsz=960,
        batch=16,
        workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",

        # Optimizer & LR
        optimizer="AdamW",
        lr0=3e-4,
        patience=120,

        # 증강
        fliplr=0.0,
        flipud=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,

        mosaic=0.3,
        mixup=0.0,

        # 결과 저장 경로
        project=r"C:\Users\ehdgu\Desktop\2people\코딩\코딩\behind_runs",
        name="exp_behind1"   # 실행할 때마다 exp_right2, exp_right3 ... 로 바꿔주세요
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
