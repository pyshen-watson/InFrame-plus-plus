import cv2
import numpy as np


WIDTH = 1920 // 2
HEIGHT = 1080 // 2



cap = cv2.VideoCapture("./yee960.mp4")
video = cv2.VideoWriter('./yee960_60.mp4', -1, 60.0, (WIDTH, HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        print("video end")
        break
    for i in range(2):
        video.write(frame)

video.release()






