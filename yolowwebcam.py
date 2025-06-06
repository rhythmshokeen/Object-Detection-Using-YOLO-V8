from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8m.pt")

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster","pen", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()

    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorC=(0, 255, 0))
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            label = f'{classNames[cls]} {conf}'
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

    fps = round(1 / (new_frame_time - prev_frame_time), 2)
    prev_frame_time = new_frame_time
    cv2.putText(img, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
