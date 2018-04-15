import cv2
import numpy as np
import pickle


# input=threshold; output=sorted contours
def contour(img, thresh):
    edges = cv2.Canny(thresh, 100, 200)
    # cv2.imshow('can', edges)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda cont: cv2.boundingRect(cont)[0])
    p = np.copy(img)
    cv2.drawContours(p, contours, -1, (0, 255, 0), 2)
    cv2.imwrite('contours.jpg', p)
    x, y, x1, y1 = [], [], [], []
    for c in contours:
        a, b, a1, b1 = cv2.boundingRect(c)
        a1, b1 = a + a1, b + b1
        x.append(a)
        y.append(b)
        x1.append(a1)
        y1.append(b1)
    i=1
    while(i<len(x)):
        xavg = min(abs(x1[i] - x[i]),abs(x1[i - 1] + x[i - 1]))
        h = max(y1[i], y1[i - 1]) - min(y[i], y[i - 1])
        if (h < xavg and (x[i]-x[i-1])<xavg):
            x[i - 1] = min(x[i - 1], x[i])
            y[i - 1] = min(y[i - 1], y[i])
            x1[i - 1] = max(x1[i - 1], x1[i])
            y1[i - 1] = max(y1[i - 1], y1[i])
            for j in range(i+1,len(x)):
                x[j-1]=x[j]
                y[j-1]=y[j]
                x1[j - 1] = x1[j]
                y1[j - 1] = y1[j]
            del x[len(x)-1]
            del y[len(y) - 1]
            del x1[len(x1) - 1]
            del y1[len(y1) - 1]
        i=i+1
    for i in range(0,len(x)):
        cv2.rectangle(img, (x[i], y[i]), (x1[i], y1[i]), (0, 255, 0), 3)
    cv2.imwrite('boxes.jpg', img)
    # cv2.imshow('boxes', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return x, y, x1, y1


# def segment(img, contours):


# img = cv2.imread('..\\e.jpg', 0)
# thresh = cv2.imread('..\\threshold.jpg', 0)
# x, y, x1, y1 = contour(img, thresh)
# f = open('boxes.pkl', 'wb')
# pickle.dump([x, y, x1, y1], f)
# f.close()
# # cv2.waitKey()
# # cv2.destroyAllWindows()
