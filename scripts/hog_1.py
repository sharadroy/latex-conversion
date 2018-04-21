import cv2
import sys
import numpy as np 
from matplotlib import pyplot as plt 
import os
import keras
import h5py
import pickle
from keras.models import model_from_json
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression as lr
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
np.set_printoptions(threshold=np.nan)

import collections
from sklearn.naive_bayes import GaussianNB as gb
latex_index= ['-', '1','2','+','X', '(',')','=','3','A','N','y','\sqrt','4','0','b','z','d','l','\sin','5','C','9','6','8','7',
    '\\times','k','\cos','T','\\alpha','e','\int','\\theta',',','f','infty','sum','\\tan','\pi']

def prework(img, flag):
    img= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=0 )
    img=cv2.resize(img,(45,45))
    img= cv2.bitwise_not(img)
    if (flag==1):
        img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        img=img.reshape(1,45,45,1)


    return img


def predictCNN(img):
    json_file = open('../model_4000.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_4000.h5")
    # print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    img=prework(img,1)

    # cv2.imshow('image',img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    pred = loaded_model.predict_classes(img, verbose=5)
    pred2 = loaded_model.predict_proba(img, verbose=5)
    clq = loaded_model.predict(img, verbose=5)

    return(pred)

def find_hog(image):
    size=image.shape
    #print(size)
    winSize = size
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True 
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,
        L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    descriptor = hog.compute(image)
    #print(descriptor.shape)
    return descriptor

img=cv2.imread('../images/1.jpg',0)
#print(img)
print(latex_index[int(predictCNN(img))])
img=prework(img,0)
hog_feature= find_hog(img)
hog_feature=hog_feature.reshape(1,hog_feature.shape[0])


sgd_model= 'sgd_model.pkl'
with open(sgd_model, 'rb') as file:  
    sgd_clf = pickle.load(file)
pred_sgd= sgd_clf.predict(hog_feature)
print(latex_index[int(pred_sgd)])

lr_model= 'lr_model.pkl'
with open(lr_model, 'rb') as file:  
    lr_clf= pickle.load(file)
pred_lr= lr_clf.predict(hog_feature)
print(latex_index[int(pred_lr)])

gnb_model= 'gnb_model.pkl'
with open(gnb_model, 'rb') as file:  
    gnb_clf = pickle.load(file)

pred_gnb= gnb_clf.predict(hog_feature)
print(latex_index[int(pred_gnb)])

dt_model= 'dt_model.pkl'
with open(gnb_model, 'rb') as file:  
    dt_clf = pickle.load(file)

pred_dt= dt_clf.predict(hog_feature)
print(latex_index[int(pred_dt)])


