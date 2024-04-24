# Samuel Lee, Aaron Pan, Abhishek Uddaraju
# CS 5330
# Spring 2024
# Sandbox for testing calorie estimation code

from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import easyocr
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Selecting imgae from a UI screen
def select_image():
    root = tk.Tk()
    root.withdraw() 

    # Open file
    file_path = filedialog.askopenfilename()
    root.destroy() 

    return file_path

# This function determines calorie of food
def calorieEstimator(image_path, model_path):
    # Initializing calorie estimate for current food and calorie per gram for each food
    current_estim = 0.0
    calorie_estimates = [('banana', 0.89),
                         ('cheesecake', 3.21),
                         ('strawberry', 0.36),
                         ('pancake', 2.27),
                         ('donut', 4.26),
                         ('pho', 0.89),
                         ('pizza', 2.82),
                         ('scallop', 1.11),
                         ('taco', 2.52),
                         ('spaghetti', 1.54)]

    # Reading image
    img = cv2.imread(image_path)

    # Setting up the YOLO model from our training & setting threshold
    model = YOLO(model_path)
    threshold = 0.5

    # Prediction
    results = model(img)[0]
    
    # Setting prediction result name for later use
    initial_result_name = ''
    
    # List to keep track of labels and scores
    labels = []
    scores = []

    # Going through prediction results and finding which is above the threshold
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if results.names[int(class_id)] != 'credit_card':
            labels.append(results.names[int(class_id)])
            scores.append(score)

    # Set result name to the label with highest score
    max_index = scores.index(max(scores))
    initial_result_name = labels[max_index]

    # Feeding image into OCR to get weight on scale
    weight = ocr_scale_weight(image_path)

    # Multiplying weight of prediction by unit calorie count
    for item in calorie_estimates:
        if initial_result_name == item[0]:
            current_estim = weight * item[1]

    
    # Returning calorie estimate
    return(initial_result_name, round(current_estim,2) , weight)

# This function performs non maxima suppresion around the box that bound the screen in case there are multiple that bound the screen at a time
def non_max_supression(results):
    boxes = []
    scores = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if results.names[int(class_id)] == 'screen':
            boxes.append([x1,y1,x2,y2])
            scores.append(score)

    boxes = np.array(boxes)
    scores = np.array(scores)

    indices = cv2.dnn.NMSBoxes(boxes[:, :4], scores, score_threshold=0.1, nms_threshold=0.)
    return boxes[indices]

# This function crops the image to only the LCD screen to feed into OCR
def crop_for_ocr(image_path, model_path):
    # Reading image
    img = cv2.imread(image_path)

    # Setting up the YOLO model from our training & setting threshold
    model = YOLO(model_path)

    # Prediction
    results = model(img)[0]
    
    # Putting through non-maxima suppression function to find the best bounding box
    box = non_max_supression(results)

    # Cropping image
    cropped_img = img[int(box[0,1]):int(box[0,3]), int(box[0,0]):int(box[0,2])]

    return(cropped_img)

# This function extracts the weight from the scale in the image using OCR
def ocr_scale_weight(image_path):
    ocr_model_path = os.path.join('..', 'runs', 'detect', 'screen_detect', 'best.pt')

    cropped_img = crop_for_ocr(image_path, ocr_model_path)
 
    cropped_img = cv2.GaussianBlur(cropped_img, (19, 19), 0)
    
    height, width = cropped_img.shape[:2]
    cropped_img = cv2.resize(cropped_img, (int(width/3), int(height/3)))
    
    height, width = cropped_img.shape[:2]

    # instance text detector
    reader = easyocr.Reader(['en'], gpu=False)
    
    # converting the image to grayscale
    gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    # inverting the grayscale image
    inverted_image = 255 - gray_image
    # Apply threshold (pixel values less than 200 become zero)
    threshold_value = 180
    _, thresholded_image = cv2.threshold(inverted_image, threshold_value, 255, cv2.THRESH_BINARY)
    # Inverting the tresholded image
    inverted_image = 255 - thresholded_image
    # converting thresholded image to 3-channel image
    thresholded_3channel = cv2.merge([inverted_image] * 3)
    # applying Gaussian blur
    thresholded_3channel = cv2.GaussianBlur(thresholded_3channel, (9, 9), 0)  # Kernel size can be adjusted


    # detect text on image
    text_ = reader.readtext(thresholded_3channel)

    threshold = 0.1
    weight = 0
    # draw bbox and text
    for t_, t in enumerate(text_):
        print(t)

        bbox, text, score = t

        if score > threshold:
            cv2.rectangle(thresholded_3channel, bbox[0], bbox[2], (0, 255, 255), 2)
            cv2.putText(thresholded_3channel, text, [bbox[0][0]+10,bbox[0][1]+20], cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255, 0), 2)
            weight = int(text)
            
    # show thresholded image
    plt.imshow(cv2.cvtColor(thresholded_3channel, cv2.COLOR_BGR2RGB))
    plt.show(block = False)

    return(weight)

def main():
    image_path = select_image()
    model_path = os.path.join('..', 'runs', 'detect', 'food_detect', 'best.pt')

    food, calories, weight = calorieEstimator(image_path, model_path)
    print("food: ", food)
    print("calories: ", calories)
    print("weight: ", weight)

if __name__ == "__main__":
    main()
