# object detection and recognition using Yolo
import cv2
from ultralytics import YOLO
import imutils
import numpy as np
import time
import torch

# Load YOLOv8 model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO('yolov8n.pt')  # Ensure you have the correct YOLOv8 model file
model.to(device)
colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype=np.uint8)
# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1000)

    if not ret:
        break

    # Perform object detection
    results = model(frame)
    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            color = [int(c) for c in colors[cls]]
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if y1 - 15 > 15:
                y = y1 - 15
            else:
                y = y1 + 15
            cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Real-Time Object Detection Using YOLOv8 by VISHVA A", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
