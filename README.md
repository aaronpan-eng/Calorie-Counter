# Calorie-Counter
### Description
This project provides an interface to track food and calories per day with a scale supplied by the user. Foods included in our database include: Cheesecake, tacos, scallops, pizza, pho, donuts, pancake, spaghetti, strawberry, and bananas. It uses YOLOv8 for object detection, which is trained on our custom dataset of 30 images per class. To track the calories, it reads the weight in grams from the LCD scale with OCR and multiplies it by grams per calorie of each food.

**For graders: We presented live during class**

### Team Members
- Aaron Pan
- Abhishek Uddaraju
- Samuel Lee

### Video demo
[![Calorie Counter Demo](https://img.youtube.com/vi/Ai11VdWSU5A/0.jpg)](https://www.youtube.com/watch?v=Ai11VdWSU5A)

### Example Images of the 10 Categories
![Screenshot 2024-04-23 192154](https://github.com/aaronpan-eng/Calorie-Counter/assets/91743944/7652ddce-0ed4-4733-9be6-4e45facfe54e)

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
```
Run main_calendar.py from src folder for the app to work properly with other files

Note: when running .py files, run them from the directory in which they are contained to make sure file paths work together
