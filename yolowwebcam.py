from ultralytics import YOLO
import cv2
import cvzone
import math
import time
# ultralytics.YOLO: For loading and running the YOLOv8 model.
# cv2: OpenCV, used for capturing video frames and image processing.
# cvzone: Simplifies OpenCV tasks (like drawing nice rectangles or text).
# math: You used it earlier for rounding, now not needed here.
# time: Used to calculate FPS (Frames Per Second).


# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height


# Load YOLOv8 model (auto-downloads weights)
model = YOLO("yolov8m.pt")  # You can use 'yolov8n.pt' for faster results on low-end PCs


#COCO classes
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


# Import Required Libraries
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
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw corner rectangle
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorC=(0, 255, 0))

            # Confidence score
            conf = round(float(box.conf[0]), 2)

            # Class ID
            cls = int(box.cls[0])

            # Put label text
            label = f'{classNames[cls]} {conf}'
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)


    # Calculate and show FPS
    fps = round(1 / (new_frame_time - prev_frame_time), 2)
    prev_frame_time = new_frame_time
    cv2.putText(img, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)


    # Show the result
    cv2.imshow("YOLOv8 Detection", img)


    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
