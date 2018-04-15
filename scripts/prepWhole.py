import cv2
import numpy as np
from skimage.morphology import skeletonize
import skimage.filters as fl

# from skimage.morphology import skeletonize

# input=whole image; output=skeletonization,threshold,edges
def prepWhole(img):
    # type: (np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]
    # cv2.imshow('C',img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    cv2.imwrite('median.jpg',img)
    # cv2.imshow('G',img)
    # img = cv2.resize(img,(45,45),interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('sz',img)
    adap = fl.threshold_minimum(img)
    # print(adap)
    temp, img = cv2.threshold(img, adap, 255, cv2.THRESH_BINARY_INV)

    # block_size = 35
    # adaptive_thresh = fl.threshold_local(img, block_size, offset=10)
    # img = 255*(img > adaptive_thresh)

    # img = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    # ret3, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('T',img)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    # thresh = img
    img = cv2.dilate(img, None, iterations=3)
    img = cv2.erode(img, None, iterations=2)
    thresh=img
    edges = cv2.Canny(thresh, 100, 200)
    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    # skel=cv2.erode(skel,element)
    # cv2.imshow("skel", skel)
    # out=skel
    # skel = cv2.dilate(skel, None, iterations=3)
    # skel = cv2.erode(skel, None, iterations=2)
    # skel = cv2.medianBlur(skel, 5)
    # temp=1*(thresh>127)
    # skel = 255*skeletonize(temp)
    return skel, thresh, edges


# img = cv2.imread('..\\e.jpg')
# out, thresh, edges = prepWhole(img)
# cv2.imshow('out', out)
# cv2.imwrite('..\\skeleton.jpg', out)
# cv2.imshow('thresh', thresh)
# cv2.imwrite('..\\threshold.jpg', thresh)
# cv2.imshow('edges', edges)
# cv2.imwrite('..\\edges.jpg', edges)
# cv2.waitKey()
# cv2.destroyAllWindows()
