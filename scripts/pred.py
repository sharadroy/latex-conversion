import cv2
import sys
import numpy as np 
from matplotlib import pyplot as plt 
import os
import keras
import h5py
from keras.models import model_from_json
def predictCNN(img):
    json_file = open('../model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    # print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # img=cv2.imread("../images/4_.jpg",0)
    img= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=0	)
    img=cv2.resize(img,(45,45))
    img= cv2.bitwise_not(img)
    img=img.reshape(1,45,45,1)
    # cv2.imshow('image',img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    pred = loaded_model.predict_classes(img, verbose=5)
    pred2 = loaded_model.predict_proba(img, verbose=5)
    clq = loaded_model.predict(img, verbose=5)

    return(pred)