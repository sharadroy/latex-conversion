import cv2
import numpy as np
from scripts import prepWhole as prep
from scripts import segment
from scripts import tree

file='e.jpg'
img=cv2.imread('..\\'+file,0)
skel,thresh,edges=prep.prepWhole(img)
cv2.imwrite('..\\skeleton.jpg', skel)
cv2.imwrite('..\\threshold.jpg', thresh)
cv2.imwrite('..\\edges.jpg', edges)
x,y,x1,y1=segment.contour(img,thresh)

for i in range(0,len(x)):
    crop=

labels=['(','2','3','+','\\int','5','x','2','-','9','\\theta',')']

start=tree.chr(index=0,label=labels[0])
prev=start
for i in range(1,len(x)):
    curr=tree.chr(index=i,label=labels[i])
    prev=tree.insert(prev,curr)

tree.printTree(start)