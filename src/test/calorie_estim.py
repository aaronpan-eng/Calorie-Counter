from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import easyocr

def calorieEstimator(image_path, model_path):
    # Initializing
    current_estim = 0.0
    calorie_estimates = [('banana', 0.89),
                         ('cheesecake', 3.21),
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
    # print(results)
    
    # Going through prediction results and finding which is above the threshold
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            initial_result_name = results.names[int(class_id)]

    # Feeding image into OCR to get weight on scale
    weight = ocr_scale_weight(image_path)

    # Multiplying weight of prediction by unit calorie count
    for item in calorie_estimates:
        if initial_result_name == item[0]:
            current_estim = weight * item[1]
            print("Food: ", initial_result_name)
            print("Calories for ", weight, " grams of ", initial_result_name, " is: ", current_estim, " calories.")
    
    # Returning calorie estimate
    return(initial_result_name, current_estim)

def crop_for_ocr(image_path, model_path):
    # Reading image
    img = cv2.imread(image_path)

    # Setting up the YOLO model from our training & setting threshold
    model = YOLO(model_path)
    threshold = 0.15

    # Prediction
    results = model(img)[0]

    # Inintializing
    cropped_img = img
    
    # Going through prediction results and finding which is above the threshold
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if results.names[int(class_id)] == 'scale':
            if score > threshold:
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

    return(cropped_img)        

def ocr_scale_weight(image_path):
    ocr_model_path = os.path.join('runs', 'detect', 'train9', 'weights', 'last.pt')

    cropped_img = crop_for_ocr(image_path, ocr_model_path)

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
    thresholded_3channel = cv2.GaussianBlur(thresholded_3channel, (5, 5), 0)  # Kernel size (5, 5) can be adjusted


    # detect text on image
    text_ = reader.readtext(thresholded_3channel)

    threshold = 0.25

    weight = 0
    # draw bbox and text
    for t_, t in enumerate(text_):
        print(t)

        bbox, text, score = t

        if score > threshold:
            cv2.rectangle(cropped_img, bbox[0], bbox[2], (0, 255, 255), 2)
            cv2.putText(cropped_img, text, [bbox[0][0]+10,bbox[0][1]+20], cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255, 0), 2)
            weight = int(text)

    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.show()

    return(weight)

def main():
    im_path = "C:/Users/aaron/Downloads/imagesdsfdsafds.jpg"
    model_path = os.path.join('runs', 'detect', 'train4', 'weights', 'last.pt')

    food, calories = calorieEstimator(im_path, model_path)

if __name__ == "__main__":
    main()
