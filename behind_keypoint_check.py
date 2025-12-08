import os, cv2, numpy as np

# ===== 경로 설정 =====
IMG_PATH   = r"C:\Users\ehdgu\Desktop\2people\코딩\코딩\image\behind_foot\behind_foot\hehind_foot (1).jpg"
LABEL_PATH = r"C:\Users\ehdgu\Desktop\2people\코딩\코딩\image\behind_foot\behind_foot\hehind_foot (1).txt"
CLASS_NAMES = ["behind_foot_1", "behind_foot_2", "behind_foot_3"]

# ===== 유틸 =====
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def _to_int_pts(norm_pts, W, H):
    pts = np.asarray(norm_pts, dtype=np.float32).reshape(-1, 2)
    return np.stack([pts[:, 0] * W, pts[:, 1] * H], axis=1).astype(np.int32)

def _parse_line(parts):
    """YOLO keypoint 라인 파싱:
       1) cls x y x y ...               (bbox 없음, v 없음)
       2) cls cx cy w h x y v ...       (bbox 있음, v 포함)
       3) cls x y v x y v ...           (bbox 없음, v 포함)
    """
    cls = int(parts[0])
    nums = list(map(float, parts[1:]))

    # bbox 포함 형태 추정: 앞 4개가 모두 [0,1] 범위
    has_bbox = len(nums) >= 4 and all(0.0 <= v <= 1.0 for v in nums[:4])
    start = 4 if has_bbox else 0
    rem = nums[start:]

    # v(가시성) 포함 여부
    if len(rem) % 3 == 0:
        xs = rem[0::3]; ys = rem[1::3]     # v는 rem[2::3]
    else:
        xs = rem[0::2]; ys = rem[1::2]

    return cls, np.column_stack([xs, ys])  # (N,2) normalized

def draw_yolo_keypoints(img, label_path, class_names):
    H, W = img.shape[:2]
    if not os.path.exists(label_path):
        return img

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 3:  # cls + 최소 1점
            continue

        cls, kp = _parse_line(parts)
        pts_px = _to_int_pts(kp, W, H)

        # 클래스별 색 고정
        np.random.seed(cls + 1234)
        color = tuple(int(x) for x in np.random.randint(50, 230, size=3))

        # 키포인트 그리기 + 인덱스 번호
        for i, (x, y) in enumerate(pts_px, start=1):
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)   # BGR = (파랑, 초록, 빨강)
            cv2.putText(img, str(i), (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)




        # 클래스 라벨
        name = class_names[cls] if 0 <= cls < len(class_names) else f"cls{cls}"
        cv2.putText(img, name, (pts_px[0][0] + 6, pts_px[0][1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
    return img

# ===== 실행 =====
if __name__ == "__main__":
    img = imread_unicode(IMG_PATH)
    vis = draw_yolo_keypoints(img, LABEL_PATH, CLASS_NAMES)

    # 미리보기 축소
    scale = 0.5
    vis_small = cv2.resize(vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale)))
    cv2.imshow("YOLO Keypoints Preview", vis_small)
    cv2.waitKey(0); cv2.destroyAllWindows()

    # 저장(원본 크기)
    save_path = os.path.splitext(IMG_PATH)[0] + "_keypoints_preview.jpg"
    cv2.imencode(".jpg", vis)[1].tofile(save_path)
    print(f"[저장 완료] {save_path}")
