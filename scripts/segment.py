import cv2
import numpy as np


# input=threshold; output=sorted contours
def contour(img, thresh):
    edges = cv2.Canny(thresh, 100, 200)
    # cv2.imshow('can', edges)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda cont: cv2.boundingRect(cont)[0])
    p=np.copy(img)
    cv2.drawContours(p,contours,-1,(0,255,0),2)
    cv2.imwrite('contours.jpg',p)
    x, y, x1, y1 = [], [], [], []
    for c in contours:
        a, b, a1, b1 = cv2.boundingRect(c)
        a1, b1 = a + a1, b + b1
        cv2.rectangle(img, (a, b), (a1,b1),(0,255,0),3)
        x.append(a)
        y.append(b)
        x1.append(a1)
        y1.append(b1)
    cv2.imwrite('boxes.jpg',img)
    cv2.imshow('boxes',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return x, y, x1, y1


# def segment(img, contours):


img = cv2.imread('d.jpg')
thresh = cv2.imread('threshold.jpg', 0)
x,y,x1,y1 = contour(img,thresh)
# cv2.waitKey()
# cv2.destroyAllWindows()
