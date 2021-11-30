import cv2
import numpy as np

WIDTH = 1920
HEIGHT = 1080

img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
for h in range(HEIGHT):
    for w in range(WIDTH):
        for i in range(3):
            img[h][w][i] = 128
#cv2.imshow('test', img)
#cv2.waitKey(0)

# FPS = 120 gray image
video = cv2.VideoWriter('./gray.mp4', -1, 120.0, (WIDTH, HEIGHT))
for t in range(240):
    video.write(img)

video.release()

cv2.destroyAllWindows()