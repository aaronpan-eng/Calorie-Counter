from ultralytics import YOLO
import os
import cv2

def calorieEstimator(image_path, model_path, weight):
    #initializing
    current_estim = 0.0
    calorie_estimates = [('banana', 0.89),
                         ('cheesecake',3.21),
                         ('strawberry', 0.36),
                         ('pancake', 2.27),
                         ('donut', 4.26),
                         ('pho', 0.89),
                         ('pizza', 2.82),
                         ('scallop', 1.11),
                         ('taco', 2.52)]

    # Reading image
    img = cv2.imread(image_path)

    # Setting up the YOLO model from our training & setting threshold
    model = YOLO(model_path)
    threshold = 0.5

    # Prediction
    results = model(img)[0]

    # Setting prediction result name for later use
    initial_result_name = ''
    
    # Going through prediction results and finding which is above the threshold
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            initial_result_name = results.names[int(class_id)]
    
    # Multiplying weight of prediction by unit calorie count
    for item in calorie_estimates:
        if initial_result_name == item[0]:
            current_estim = weight * item[1]
            print(current_estim)
    
    # Returning calorie estimate
    return(current_estim)

    
