import cv2
import numpy as np

img = cv2.imread('b.jpg')
thresh = cv2.imread('threshold.jpg', 0)
temp=1*(thresh>127)
# thresh=cv2.GaussianBlur(thresh,(3,3),0)
edges = cv2.Canny(thresh, 100, 200)
cv2.imshow('can', edges)
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda cont: cv2.boundingRect(cont)[0])
# new=cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('con', img)
cv2.waitKey()
cv2.destroyAllWindows()
