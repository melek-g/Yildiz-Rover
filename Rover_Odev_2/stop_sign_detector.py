import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import math 


def show_images(images: list):
    for i, img in enumerate(images):
        cv.imshow(f'Image {i+1}', img)
        cv.waitKey(0)
        cv.destroyWindow(f'Image {i+1}')

# OKUNACAK RESIM ICIN " " ICINE KLASORDEN PATH YAPISTIRINIZ.
img = cv.imread("C:/Users/berko/Desktop/ROVER_ODEVLER/Rover_Odev_2/stop_sign_dataset/2.jpg")

downscale = cv.resize(img, (img.shape[1]//2, img.shape[0]//2), cv.INTER_AREA)

hsv = cv.cvtColor(downscale, cv.COLOR_BGR2HSV)
h,s,v = cv.split(hsv)


mask1 = cv.inRange(hsv, np.array([0, 100, 10]), np.array([10, 255, 255]))
mask2 = cv.inRange(hsv, np.array([160, 100, 10]), np.array([180, 255, 255]))
mask = cv.bitwise_or(mask1, mask2)

images = [downscale, hsv, h, s, v, mask]

# Ara basamaklarin ciktilarini goruntulemek icin bir alt satirdaki comment kaldirilarak kod calistirilabilir: 
show_images(images) 

contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

rects = []
for cnt in contours:
    a = cv.contourArea(cnt)
    if a >= 1700:
        rects.append(cv.boundingRect(cnt)) 

keep = [True] * len(rects)
for i, (x1, y1, w1, h1) in enumerate(rects):
    for j, (x2, y2, w2, h2) in enumerate(rects):
        if i == j:
            continue

        if (x1 <= x2 and y1 <= y2 and
            x1 + w1 >= x2 + w2 and
            y1 + h1 >= y2 + h2):
            keep[j] = False

for k, ok in enumerate(keep):
    if ok:
        x, y, w, h = rects[k]
        cv.rectangle(downscale, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f'The center of the detected sign is at: ({x + w//2}, {y + h//2})')


cv.imshow('Detected Stop Sign(s)', downscale)
cv.waitKey(0)
cv.destroyAllWindows() 