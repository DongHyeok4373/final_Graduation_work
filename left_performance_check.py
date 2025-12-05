from ultralytics import YOLO
import cv2

def test_left_foot():
    # 학습된 포즈 모델 불러오기
    model = YOLO(r"C:\Users\ehdgu\Desktop\2people\코딩\코딩\left_runs\exp_left17\weights\best.pt")

    # 이미지 추론
    results = model.predict(
        source=r"C:\Users\ehdgu\Desktop\참고\left_foot (3).JPG",
        imgsz=640,
        save=True   # 결과 시각화 저장 (runs/pose/predict 폴더)
    )

    for r in results:
        # 점 크기를 키워서 시각화
        im = r.plot(kpt_radius=10)

        # 창 크기 축소 (예: 가로/세로 50%로 줄이기)
        h, w = im.shape[:2]
        im_small = cv2.resize(im, (w // 2, h // 2))

        # 화면에 띄우기
        cv2.imshow("Prediction (Resized)", im_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 좌표/신뢰도 출력
        print(r.keypoints.xy)    # keypoints 좌표 (픽셀 단위)
        print(r.keypoints.conf)  # keypoints confidence

if __name__ == "__main__":
    test_left_foot()
