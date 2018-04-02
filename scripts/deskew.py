import cv2
import sys
import numpy as np 
from matplotlib import pyplot as plt 
import os
import tensorflow as tf 

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    (SZ,SZ)=img.shape
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

# gray=cv2.imread("test.jpg",0)
# #gray=cv2.resize(gray,(350,350))
# #image=cv2.GaussianBlur(gray,(3,3),0.1)
# ret,image=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
# gray_display=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
# #print(contrs)

# # size = np.size(image)
# # skel = np.zeros(image.shape, np.uint8)
# # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# # done = False

# # while (not done):
# #     eroded = cv2.erode(image, element)
# #     temp = cv2.dilate(eroded, element)
# #     temp = cv2.subtract(image, temp)
# #     skel = cv2.bitwise_or(skel, temp)
# #     image = eroded.copy()

# #     zeros = size - cv2.countNonZero(image)
# #     if zeros == size:
# #         done = True
# #cv2.drawContours(gray_display,contour,-1,(0,0,255),5)

# image,contour, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# for individual_contour in contour:

# 	x,y,w,h=cv2.boundingRect(individual_contour)
# 	cv2.rectangle(gray_display,(x,y),(x+w,y+h),(0,0,255),2)



# print(len(contour),gray_display.shape)
# plt.imshow(gray_display)
# plt.show()
img=cv2.imread('g.jpg',0)
img=cv2.resize(img,(300,300))
ret,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
img_new=deskew(img)
#plt.subplot(2,1,1)
#plt.imshow(img,cmap='gray')
#plt.subplot(2,1,2)
plt.imshow(img_new,cmap='gray')
plt.show()
