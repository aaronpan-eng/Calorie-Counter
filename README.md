# Calorie-Counter
This project provides an interface to track food and calories per day with a scale supplied by the user. Foods included in our database include: Cheesecake, tacos, scallops, pizza, pho, donuts, pancake, spaghetti, strawberry, and bananas. It uses YOLOv8 for object detection, which is trained on our custom dataset of 30 images per class. To track the calories, it reads the weight in grams from the LCD scale with OCR and multiplies it by grams per calorie of each food.

#### Team Members
- Aaron Pan
- Abhishek Uddaraju

## Installation/Instructions 

### 1. Cloning the project
```
$ git clone https://github.com/aaronpan-eng/Calorie-Counter.git
$ cd /project location
```

### 2. Setup Visual Studio Code

Visual Studio Code is a lightweight but powerful source code editor that runs on your desktop.

1. [Download](https://code.visualstudio.com/) and install Visual Studio Code.

### 3. Command line arguments required to run the Executable
```
$ cd /<project_location>/src

Run main_calendar.py from src folder for it to work properly with other files
