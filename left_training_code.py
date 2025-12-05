from ultralytics import YOLO
import torch

def main():
    # Pretrained pose model 불러오기
    model = YOLO(r"C:\Users\ehdgu\Desktop\2people\코딩\코딩\left_runs\exp_left119\weights\best.pt") 

    # left_foot 전용 데이터셋 yaml
    data_yaml = r"C:\Users\ehdgu\Desktop\2people\코딩\코딩\foot_dataset_updated\left_view\data.yaml"

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

        # 결과 저장 경로 (분리해서 관리)
        project=r"C:\Users\ehdgu\Desktop\2people\코딩\코딩\left_runs",
        name="exp_left1"   # 실행할 때마다 exp_left2, exp_left3 ... 이렇게 바꿔주면 됨
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
