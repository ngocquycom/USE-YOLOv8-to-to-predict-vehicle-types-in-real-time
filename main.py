import cv2
from ultralytics import YOLO
import numpy as np
import time

VIDEO_PATH = "../images/video1.mp4"
MODEL_PATH = "best_done.pt"  
OUTPUT_PATH = "result.mp4"

LANES = {
    "Lane1": [(663,167), (768,169), (682,508), (461,503)],
    "Lane2": [(768,169), (882,172), (921,512), (682,510)],
    "Lane3": [(885,172), (987,176), (1166,520), (921,515)],
    "Lane4": [(987,176), (1087,179), (1383,520), (1171,520)],
    "Lane5": [(1085,176), (1187,179), (1585,520), (1385,520)]
}

# ĐỊNH NGHĨA MÀU SẮC RIÊNG CHO TỪNG LÀN (BGR)
LANE_COLORS = {
    "Lane1": (0, 0, 255),    # Đỏ (Red)
    "Lane2": (0, 255, 0),    # Xanh Lá (Green)
    "Lane3": (255, 0, 0),    # Xanh Dương (Blue)
    "Lane4": (0, 255, 255),  # Vàng (Yellow)
    "Lane5": (255, 0, 255)  # Hồng/Tím (Magenta)
}

CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
CONF_THRESHOLD = 0.35
TRACKER = "bytetrack.yaml" 

def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x_i, y_i = poly[i]
        x_j, y_j = poly[(i + 1) % n]
        intersect = ((y_i > y) != (y_j > y)) and \
                    (x < (x_j - x_i) * (y - y_i) / (y_j - y_i + 1e-9) + x_i)
        if intersect:
            inside = not inside
    return inside

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO_PATH}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

# THAY ĐỔI: Thêm chiều cao cho khu vực hiển thị thông tin
INFO_HEIGHT = 60
NEW_H = h + INFO_HEIGHT 
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, NEW_H))

global_seen = set()                 # all track ids seen (unique vehicles)
lane_seen = {k: set() for k in LANES.keys()}   # per-lane seen IDs
lane_counts = {k: 0 for k in LANES.keys()}     # per-lane unique counts

frame_no = 0
t0 = time.time()

print("[INFO] Start processing... (press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1
    results = model.track(source=frame, persist=True, tracker=TRACKER, conf=CONF_THRESHOLD)

    # THAY ĐỔI: Tạo khung hình mới có nền đen ở dưới
    disp = np.zeros((NEW_H, w, 3), dtype=np.uint8)

    if len(results) == 0:
        out_frame = frame
    else:
        res = results[0]
        out_frame = res.plot() # Khung hình đã có bounding box và ID
        disp[:h, :] = out_frame # Đặt khung hình YOLO plot lên trên

        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            for i in range(len(boxes)):
                try:
                    xyxy = boxes[i].xyxy[0].cpu().numpy() if hasattr(boxes[i].xyxy[0], "cpu") else np.array(boxes[i].xyxy[0])
                except Exception:
                    try:
                        xyxy = np.array(boxes[i].xyxy[0])
                    except Exception:
                        continue

                x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                try:
                    cls = int(boxes[i].cls[0].item())
                except Exception:
                    cls = None

                try:
                    tid = int(boxes[i].id[0].item())
                except Exception:
                    tid = None
                
                if cls is None:
                    continue

                for lane_name, poly in LANES.items():
                    if point_in_poly(cx, cy, poly):
                        if tid is not None:
                            if tid not in lane_seen[lane_name]:
                                lane_seen[lane_name].add(tid)
                                lane_counts[lane_name] = len(lane_seen[lane_name])
                            if tid not in global_seen:
                                global_seen.add(tid)
                        else:
                            pass

    for lane_name, poly in LANES.items():
        pts = np.array(poly, np.int32).reshape((-1,1,2))
        color = LANE_COLORS.get(lane_name, (0, 200, 255))
        cv2.polylines(disp, [pts], True, color, 2)

    total_unique = len(global_seen)
    sum_of_lanes = sum(lane_counts.values())

    cv2.putText(disp, f"Total unique vehicles: {total_unique}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(disp, f"Sum lanes (may double count): {sum_of_lanes}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # Đặt bảng thông tin lên NGAY DƯỚI dòng Total (vùng trên), tránh góc bị che
    info_y_start = 110  # cố định dưới phần total
    num_lanes = len(LANES)
    column_width = w // num_lanes 
    
    vertical_y = info_y_start
    for i, (lane_name, count) in enumerate(lane_counts.items()):
        color = LANE_COLORS.get(lane_name, (255, 255, 255))
        text = f"{lane_name}: {count}"

        cv2.putText(
            disp,
            text,
            (30, vertical_y),  # canh trái
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )
        vertical_y += 30  # xuống dòng()):
        color = LANE_COLORS.get(lane_name, (255, 255, 255))
        text = f"{lane_name}: {count}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        current_x = (column_width * i) + (column_width - text_size[0]) // 2 
        
        cv2.putText(
            disp,
            text,
            (current_x, info_y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, 
            color,
            2, 
            cv2.LINE_AA
        )

    writer.write(disp)
    cv2.imshow("Traffic Counter", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

t1 = time.time()
print("[INFO] Done. Total frames:", frame_no, "Elapsed:", round(t1-t0,2), "s")
print("[INFO] Total unique vehicles:", total_unique)
print("[INFO] Lane counts:", lane_counts)
print("Đã xem xong video rồi nhé! ")
print("--------------------------------------------🚗🚙🚌🚚-----------------------------------------------")