# NAMES
# CS 5330
# Spring 2024
# DESCRIPTION

# import statements
import os
import json
import time
import random
from shutil import copyfile
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


# This function determines if a food item is identifiable in the network or not
def identified_food():
    return True


# This function determines if the weight of the food is found in the image
def identified_weight():
    return True


# This function adds the food name, food image file name, weight, calories, and total calories to the database
def add_food_to_database(window, name, file_path, weight, calories, current_date):
    # get directory where imagesd will be stored for the current date
    base_directory = os.path.join("saved_food_images", current_date.replace("-", "/"))
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
    food_add_window, name, file_path, weight, calories, current_date
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
            food_add_window, name, file_path, weight, calories, current_date
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
def add_food(current_date):
    # select image in file dialog
    file_path = filedialog.askopenfilename(
        title="Select an Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )

    # if no valid file selected, exit function
    if not file_path:
        tk.messagebox.showerror("No File Selected", "Please select a valid image file.")
        return

    # TODO: GET ACTUAL VALUES FROM OTHER PARTS
    # if valid file, check if identifiable food
    if identified_food():
        # check if weight is found
        if identified_weight():
            # open new window
            food_add_window = create_food_add_window()
            customize_food_add_window(
                food_add_window, "banana", file_path, "100g", 155, current_date
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
    json_data_path = os.path.join("calories.json")
    if os.path.exists(json_data_path):
        with open(json_data_path, "r") as file:
            data = json.load(file)

    # default value
    calories = 0

    # if calories for date exist
    calories = data[selected_date]["total_calories"]

    # date label
    date_label = tk.Label(day_window, text=f"Date: {selected_date}")
    date_label.grid(row=0, column=0, padx=20, pady=20)

    # calorie label
    cal_label = tk.Label(day_window, text=f"Calories: {calories}")
    cal_label.grid(row=1, column=0, padx=20, pady=20)

    # button to add food
    add_button = tk.Button(
        day_window, text="Add Food", command=lambda: add_food(selected_date)
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


# This function opens the final individual day window
def individual_day_window(selected_date):
    day_window = create_day_window(selected_date)
    customize_day_window(day_window, selected_date)
