# Samuel Lee, Aaron Pan, Abhishek Uddaraju
# CS 5330
# Spring 2024
# DESCRIPTION

# import statements
import sys
import tkinter as tk
from tkinter import ttk
from tkcalendar import Calendar
import individual_day


# This function resizes the window to be a square
def resize_square(event):
    # access root via event
    root = event.widget

    # only resize if it's a window/root
    if isinstance(root, tk.Tk):
        width = root.winfo_width()
        height = root.winfo_height()

        # resize only if not a square
        if width != height:
            size = max(root.winfo_width(), root.winfo_height())
            root.geometry(f"{size}x{size}")


# This function handles the event of date selection on the calendar
def on_date_select(event, cal, date_label):
    selected_date = cal.get_date()
    date_label.config(text=f"Selected Date: {selected_date}")


# This function creates the root window and its settings
def createRoot():
    print("Creating root window...")
    # create root and set basic properties
    root = tk.Tk()
    root.geometry("600x600")
    root.title("Calorie Tracker")
    root.configure(bg="#fff8e6")

    # set style of window
    style = ttk.Style(root)
    style.theme_use("clam")

    # bind the resize_square function to Configure
    root.bind("<Configure>", resize_square)

    # configure row and column to expand or contract as window size changes
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    return root


# This function creates elements within the root and customizes them
def customizeRoot(root):
    print("Creating root elements...")
    # place calendar in the grid
    this_cal = Calendar(
        root,
        setmode="day",
        date_pattern="mm/dd/yyyy",
        showweeknumbers=False,
        showothermonthdays=False,
        firstweekday="sunday",
        bordercolor="#563517",
        background="#6f4827",
        foreground="#efe3d1",
        normalforeground="#6F4E37",
        normalbackground="#fff6ec",
        weekendforeground="#6F4E37",
        weekendbackground="#fff6ec",
        selectforeground="#efe3d1",
        selectbackground="#3d240b",
    )
    # add date selected label
    date_label = tk.Label(root, text="Select a date...", font=("Helvetica", 12))
    date_label.grid(row=1, column=0, padx=40, pady=40)

    # this enables the calendar to stick to all 4 directions
    this_cal.grid(row=0, column=0, sticky="news")

    # these define event handlers for the calendar
    this_cal.bind(
        "<<CalendarSelected>>",
        lambda event, cal=this_cal, date_label=date_label: on_date_select(
            event, cal, date_label
        ),
    )

    # button to click to open a specific date
    open_button = tk.Button(
        root,
        text="Open",
        command=lambda: individual_day.individual_day_window(this_cal.get_date()),
    )
    open_button.grid(row=2, column=0, pady=20)

    # add current calorie count label


# main function
def main(argv):
    # creates and customizes root as grid
    root = createRoot()

    # creates elements within root
    customizeRoot(root)

    # enters loop that waits for feedback
    root.mainloop()

    return


if __name__ == "__main__":
    main(sys.argv)
