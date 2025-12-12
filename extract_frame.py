import cv2

VIDEO_PATH = r"C:\Users\ADMIN\Downloads\nguyên lý máy học\images\video1.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if ret:
    cv2.imwrite("frame.jpg", frame)
    print("Đã lưu frame.jpg")
else:
    print("Không đọc được video")

cap.release()
