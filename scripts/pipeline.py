import cv2
import sys
import numpy as np
from scripts import prepWhole as prep
from scripts import segment
from scripts import tree
from scripts import pred

if len(sys.argv)<2:
    file='n.jpg'
else:
    file=sys.argv[1]
img=cv2.imread('..\\'+file)
copy=img.copy()
skel,thresh,edges=prep.prepWhole(img)
cv2.imwrite('..\\skeleton.jpg', skel)
cv2.imwrite('..\\threshold.jpg', thresh)
cv2.imwrite('..\\edges.jpg', edges)
x,y,x1,y1=segment.contour(img,thresh)

labels=[]

for i in range(0,len(x)):
    crop=thresh[y[i]:y1[i],x[i]:x1[i]]
    dif_abs = abs(int(x1[i]-x[i] - y1[i]+y[i]))
    padding = int(dif_abs / 2)
    if (x1[i]-x[i]) > (y1[i]-y[i]):
        padded_img = cv2.copyMakeBorder(crop, padding, dif_abs - padding, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        padded_img = cv2.copyMakeBorder(crop, 0, 0, padding, dif_abs - padding, cv2.BORDER_CONSTANT, value=0)
    size45=cv2.resize(padded_img, (45, 45))
    thin=prep.thinning(1*(size45>125))
    thin=thin.astype('uint8')
    # thin = cv2.resize(thin*255, (45, 45))
    # thin = cv2.bitwise_not(thin)
    new=thin*255
    # cv2.imshow('c',new)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    labels.append(pred.predictCNN(thin))


# labels=['(','2','3','+','\\int','5','x','2','-','9','\\theta',')']

start=tree.chr(index=0,label=labels[0])
prev=start
for i in range(1,len(x)):
    curr=tree.chr(index=i,label=labels[i])
    prev=tree.insert(prev,curr,x,x1,y,y1)

tree.printTree(start)