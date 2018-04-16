import cv2
import sys
import numpy as np 
from matplotlib import pyplot as plt 
import os
import keras
import h5py
from keras.models import model_from_json
def predictCNN(img):
    json_file = open('..\\model_4000_binary1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_4000_binary1.h5")
    # print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # img=cv2.imread("../images/4_.jpg",0)
    # img= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=0	)
    # img=cv2.resize(img,(45,45))
    # img= cv2.bitwise_not(img)
    # cv2.imshow('image', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    img=img.reshape(1,45,45,1)

    pred = loaded_model.predict_classes(img, verbose=5)
    pred2 = loaded_model.predict_proba(img, verbose=5)
    clq = loaded_model.predict(img, verbose=5)
    data_values = ['-', '1', '2', '+', 'x', '(', ')', '=', '3', 'a', 'n', 'y', '\\sqrt', '4', '0', 'b', 'z', 'd', 'l',
                   '\\sin', '5', 'c', '9', '6', '8', '7',
                   '\\times', 'k', '\\cos', 't', '\\alpha', 'e', '\\int', '\\theta', ',', 'f', '\\infty', '\\Sigma', '\\tan', '\\pi']
    return data_values[int(pred)]