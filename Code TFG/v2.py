import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.interpolate import interp1d
from matplotlib import pyplot

import tkinter as tk
from tkinter import filedialog
import os

class ExerciseSelector:
    def __init__(self, master):
        self.master = master
        master.title("Exercise Selector")

        self.label = tk.Label(master, text="Select the exercise you want to perform:")
        self.label.pack()

        self.squat_button = tk.Button(master, text="Squat", command=lambda: self.video_selector("Squat"))
        self.squat_button.pack(pady=5)

        self.bench_button = tk.Button(master, text="Bench Press", command=lambda: self.video_selector("Bench Press"))
        self.bench_button.pack(pady=5)

        self.deadlift_button = tk.Button(master, text="Deadlift", command=lambda: self.video_selector("Deadlift"))
        self.deadlift_button.pack(pady=5)

        self.close_button = tk.Button(master, text="Close", command=master.quit)
        self.close_button.pack(pady=10)

    def video_selector(self, exercise):
        if exercise == "Squat":
            print("Selected Squat exercise. Executing Squat-specific code...")

            # Execute SQ.py
            mycode_path = os.path.abspath("SQ.py")
            with open(mycode_path, "r") as f:
                mycode_text = f.read()
                
            exec(mycode_text)

        if exercise == "Bench Press":
            print("Selected Bench Press exercise. Executing Bench Press-specific code...")

            # Execute B.py
            mycode_path = os.path.abspath("B.py")
            with open(mycode_path, "r") as f:
                mycode_text = f.read()
                
            exec(mycode_text)

        if exercise == "Deadlift":
            print("Selected Deadlift exercise. Executing Deadlift-specific code...")

            # Execute DL.py
            mycode_path = os.path.abspath("DL.py")
            with open(mycode_path, "r") as f:
                mycode_text = f.read()
                
            exec(mycode_text)


root = tk.Tk()
my_gui = ExerciseSelector(root)
root.mainloop()
