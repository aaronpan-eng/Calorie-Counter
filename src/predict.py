from ultralytics import YOLO
from calorie_estim import calorieEstimator
import os
import cv2

model_path = os.path.join('runs', 'detect', 'train4', 'weights', 'last.pt')
model = YOLO(model_path)
threshold = 0.5

video_capture = cv2.VideoCapture(1)

if not video_capture.isOpened:
    print("video capture dev is not open")
    exit()

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("err: no frame")

    results = model(frame)[0]
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1)

    if key == 'q':
        break