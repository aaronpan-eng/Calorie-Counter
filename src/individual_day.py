# Samuel Lee, Aaron Pan, Abhishek Uddaraju
# CS 5330
# Spring 2024
# DESCRIPTION TODO: add description

# import statements
import os
import cv2
import json
import time
import random
from ultralytics import YOLO
from shutil import copyfile
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import easyocr


# This function determines if a food item is identifiable in the network or not
def identified_food(image_path, model_path):
    img = cv2.imread(image_path)

    # Setting up the YOLO model from our training & setting threshold
    model = YOLO(model_path)
    threshold = 0.5

    # Prediction
    results = model(img)[0]
    
    # Going through prediction results and finding which is above the threshold
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if results.names[int(class_id)] != 'screen':
            return True
        elif results.names[int(class_id)] == None:
            return False
    return False


# This function determines if the weight of the food is found in the image
def identified_weight(image_path, model_path):
    img = cv2.imread(image_path)

    # Setting up the YOLO model from our training & setting threshold
    model = YOLO(model_path)
    threshold = 0.5

    # Prediction
    results = model(img)[0]
    
    # Going through prediction results and finding which is above the threshold
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if results.names[int(class_id)] == 'screen':
            return True
    
    return False

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
                         ('sphagetti', 1.54)]

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
        counter = 0
        if results.names[int(class_id)] != 'credit_card':
            if (counter == 0):
                initial_result_name = results.names[int(class_id)]
                # print(initial_result_name)
                # TODO: can also store the detected foods in a list and have the user choose between the unique items in the list if the first detected food isnt right
                # alternatively could use non-maxima suppression to only return the food with the most occurance and highest probability

    # Feeding image into OCR to get weight on scale
    weight = ocr_scale_weight(image_path)

    # Multiplying weight of prediction by unit calorie count
    for item in calorie_estimates:
        if initial_result_name == item[0]:
            current_estim = weight * item[1]
            # print("Food: ", initial_result_name)
            # print("Calories for ", weight, " grams of ", initial_result_name, " is: ", current_estim, " calories.")
    
    # Returning calorie estimate
    return(initial_result_name, current_estim, weight)

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
    # rows, cols, _ = img.shape

    # print("Rows:", rows)
    # print("Columns:", cols)

    # Setting up the YOLO model from our training & setting threshold
    model = YOLO(model_path)

    # Prediction
    results = model(img)[0]
    
    # Putting through non-maxima suppression function to find the best bounding box
    box = non_max_supression(results)

    # Cropping image
    cropped_img = img[int(box[0,1]):int(box[0,3]), int(box[0,0]):int(box[0,2])]

    return(cropped_img)
# TODO: add error check for crop to make sure that it only stores the "screen" boxes into the array for non-maximum supression

# This function extracts the weight from the scale in the image using OCR
def ocr_scale_weight(image_path):
    ocr_model_path = os.path.join('runs', 'detect', 'train8', 'weights', 'best.pt')

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
            

    print("weight: ", weight)
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.show(block = False)

    return(weight)

# This function adds the food name, food image file name, weight, calories, and total calories to the database
def add_food_to_database(
    window, name, file_path, weight, calories, current_date, day_window
):
    # get directory where imagesd will be stored for the current date
    base_directory = os.path.join("..", "data", "saved_food_images", current_date.replace("/", "-"))
    os.makedirs(base_directory, exist_ok=True)

    # generate random file name to save to prevent duplicates
    timestamp = int(time.time())
    random_int = random.randint(1, 10000)
    file_extension = os.path.splitext(file_path)[-1]
    new_file_name = f"{timestamp}_{random_int}{file_extension}"
    new_file_path = os.path.join(base_directory, new_file_name)

    # copy image to data location
    copyfile(file_path, new_file_path)
    data = {}

    # specify JSON file location and load data if it exists
    json_data_path = os.path.join("calories.json")
    if os.path.exists(json_data_path):
        with open(json_data_path, "r") as file:
            data = json.load(file)

    # if current date does not exist in json database
    if current_date not in data:
        data[current_date] = {"entries": [], "total_calories": 0}

    # append new entry into json
    transaction = {
        "name": name,
        "image_file_path": new_file_path,
        "weight": weight,
        "calories": calories,
    }

    data[current_date]["entries"].append(transaction)
    data[current_date]["total_calories"] += calories

    # update data onto json file
    with open(json_data_path, "w") as file:
        json.dump(data, file, indent=4)

    # close window
    close_current_window(window)

    # update day window
    customize_day_window(day_window, current_date)


# This function deletes the entry from the database and the folder of saved images
def delete_food_from_database(del_entry, day_window, current_date):
    # get calories value of entry to delete
    del_calorie = del_entry["calories"]

    # delete image file from stored images
    if os.path.exists(del_entry["image_file_path"]):
        os.remove(del_entry["image_file_path"])

    # load JSON file
    json_data_path = os.path.join("calories.json")
    if os.path.exists(json_data_path):
        with open(json_data_path, "r") as file:
            data = json.load(file)
    else:
        print("JSON database not found!")
        return

    with open(json_data_path, "r+") as file:
        data = json.load(file)

        if current_date in data:
            old_entries = data[current_date]["entries"]
            new_entries = [
                entry
                for entry in old_entries
                if entry["image_file_path"] != del_entry["image_file_path"]
            ]

            # replace old entries with new entries
            data[current_date]["entries"] = new_entries

            # replace old calories count with new calories count
            data[current_date]["total_calories"] -= del_calorie

            # write database to JSON after updating
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()

    # update day window
    customize_day_window(day_window, current_date)


# This function closes the current window
def close_current_window(window):
    window.destroy()


# This function creates and opens the food add window
def create_food_add_window():
    # create new window
    food_window = tk.Toplevel()
    food_window.title("Add Food")
    food_window.geometry("600x600")
    food_window.grid_columnconfigure(0, weight=1)

    return food_window


# This function customizes the food add window
def customize_food_add_window(
    food_add_window, name, file_path, weight, calories, current_date, day_window
):
    # date label
    food_name_label = tk.Label(food_add_window, text=f"Name: {name}")
    food_name_label.grid(row=0, column=0, padx=20, pady=20)

    # image label
    image_pil = Image.open(file_path)
    image_pil = image_pil.resize((200, 200), Image.Resampling.LANCZOS)
    image_tk = ImageTk.PhotoImage(image_pil)

    image_label = tk.Label(food_add_window, image=image_tk)
    image_label.image = image_tk
    image_label.grid(row=1, column=0, padx=20, pady=20)

    # calorie label
    weight_label = tk.Label(food_add_window, text=f"Weight: {weight}")
    weight_label.grid(row=2, column=0, padx=20, pady=20)

    # calorie label
    cal_label = tk.Label(food_add_window, text=f"Calories: {calories}")
    cal_label.grid(row=3, column=0, padx=20, pady=20)

    # button to submit food to day
    add_button = tk.Button(
        food_add_window,
        text="Add Food To Day",
        command=lambda: add_food_to_database(
            food_add_window, name, file_path, weight, calories, current_date, day_window
        ),
    )
    add_button.grid(row=4, column=0, padx=20, pady=20)

    # button to cancel
    cancel_button = tk.Button(
        food_add_window,
        text="Cancel",
        command=lambda: close_current_window(food_add_window),
    )
    cancel_button.grid(row=5, column=0, padx=20, pady=20)


# This function asks user for an image and checks it validity
def add_food(current_date, day_window):
    # select image in file dialog
    file_path = filedialog.askopenfilename(
        title="Select an Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )

    # if no valid file selected, exit function
    if not file_path:
        tk.messagebox.showerror("No File Selected", "Please select a valid image file.")
        return

    # model parameters for detecting food for calorie estimate and screen for weight estimate
    current_directory = os.getcwd()
    food_model_path = os.path.join(current_directory, 'runs', 'detect', 'train4', 'weights', 'best.pt')
    scale_model_path = os.path.join(current_directory, 'runs', 'detect', 'train8', 'weights', 'best.pt')

    # TODO: GET ACTUAL VALUES FROM OTHER PARTS
    # if valid file, check if identifiable food
    if identified_food(file_path, food_model_path):
        
        # check if weight is found
        if identified_weight(file_path, scale_model_path):
            food, calories, weight = calorieEstimator(file_path, food_model_path)
            print("food: ", food)
            print("calories: ", calories)
            print("weight", weight)

            # open new window
            food_add_window = create_food_add_window()
            customize_food_add_window(
                food_add_window,
                food,
                file_path,
                weight,
                calories,
                current_date,
                day_window,
            )

        else:
            tk.messagebox.showerror(
                "No Weight Found", "There is no weight found in the image."
            )
        return
    else:
        # non-identifiable food
        tk.messagebox.showerror(
            "Unidentified food", "The image has no identifiable food."
        )
        return

    # if identifiable, show image, food name, weight, and calories


# This function creates and opens the individual day window
def create_day_window(selected_date):
    # create new window
    day_window = tk.Toplevel()
    day_window.title(f"Calorie Details: {selected_date}")
    day_window.geometry("600x600")
    day_window.grid_columnconfigure(0, weight=1)
    day_window.grid_rowconfigure(3, weight=1)

    return day_window


# This function customizes the individual day window
def customize_day_window(day_window, selected_date):
    # get data from json
    data = []
    json_data_path = os.path.join("calories.json")
    if os.path.exists(json_data_path):
        with open(json_data_path, "r") as file:
            data = json.load(file)

    # default value
    calories = 0

    # if calories for date exist
    if selected_date in data:
        calories = data[selected_date]["total_calories"]

    # date label
    date_label = tk.Label(day_window, text=f"Date: {selected_date}")
    date_label.grid(row=0, column=0, padx=20, pady=20)

    # calorie label
    cal_label = tk.Label(day_window, text=f"Calories: {calories}")
    cal_label.grid(row=1, column=0, padx=20, pady=20)

    # button to add food
    add_button = tk.Button(
        day_window, text="Add Food", command=lambda: add_food(selected_date, day_window)
    )
    add_button.grid(row=2, column=0, padx=20, pady=20)

    # create canvas and scrollbar and tie the original calorie table frame with canvas
    canvas = tk.Canvas(day_window, borderwidth=0)
    calorie_table_frame = tk.Frame(canvas, padx=20, pady=20)
    scrollbar = tk.Scrollbar(day_window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.grid(row=3, column=1, sticky="ns")
    canvas.grid(row=3, column=0, sticky="nsew")
    canvas.create_window((0, 0), window=calorie_table_frame, anchor="nw")

    # re-configure scroll region whenever there is a configuration change
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    # re-configure width of canvas dynamically
    def on_canvas_configure(event):
        canvas_width = event.width
        canvas.itemconfig(1, width=canvas_width)

    # set handlers/listeners
    calorie_table_frame.bind("<Configure>", on_frame_configure)
    canvas.bind("<Configure>", on_canvas_configure)

    # set the headers
    table_headers = ["Pic", "Food", "Weight", "Calories"]
    for index, header in enumerate(table_headers):
        header_label = tk.Label(
            calorie_table_frame,
            text=header,
            bg="lightgray",
            font=("Helvetica", 12, "bold"),
        )
        header_label.grid(row=0, column=index, sticky="ew")

    # edit column configuration
    for index in range(len(table_headers)):
        calorie_table_frame.grid_columnconfigure(index, weight=1)

    # populate with rows from database
    entries = []
    if selected_date in data:
        entries = data[selected_date]["entries"]
    for row_index, entry in enumerate(entries, start=1):
        # populate image
        image_pil = Image.open(entry.get("image_file_path"))
        image_pil = image_pil.resize((200, 200), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(image_pil)
        image_label = tk.Label(calorie_table_frame, image=image_tk)
        image_label.grid(row=row_index, column=0)
        image_label.image = image_tk
        # populate text
        tk.Label(calorie_table_frame, text=entry.get("name")).grid(
            row=row_index, column=1
        )
        tk.Label(calorie_table_frame, text=entry.get("weight")).grid(
            row=row_index, column=2
        )
        tk.Label(calorie_table_frame, text=entry.get("calories")).grid(
            row=row_index, column=3
        )
        # create button to delete if needed
        tk.Button(
            calorie_table_frame,
            text="X",
            command=lambda: delete_food_from_database(entry, day_window, selected_date),
        ).grid(row=row_index, column=4)


# This function opens the final individual day window
def individual_day_window(selected_date):
    day_window = create_day_window(selected_date)
    customize_day_window(day_window, selected_date)
