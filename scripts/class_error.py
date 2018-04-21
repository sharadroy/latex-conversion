import cv2
import sys
import numpy as np 
from matplotlib import pyplot as plt 
import os
import keras
import h5py
from keras.models import model_from_json

json_file = open('..\\model_4000_binary1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_4000_binary1.h5")
# print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
path= "../extracted_images/"

files=os.listdir(path)
print(files)
data_values=['-', '1','2','+','X', '(',')','=','3','A','N','y','sqrt','4','0','b','z','d','l','sin','5','C','9','6','8','7',
'times','k','cos','T','alpha','e','int','theta',',','f','infty','sum','tan','pi']
img= cv2.imread("../images/4.jpg",0)
for label_idx, i in enumerate(data_values):
    file_path=os.path.join(path,i)
    new_files=os.listdir(file_path)
    cnt=0
    count=0
    for idx, j in enumerate(new_files):
        img=cv2.imread(os.path.join(file_path,j),0)
        img= 1*(img>127)
        img=img.reshape(1,45,45,1)
        # img=img.astype('uint8')
        pred = loaded_model.predict_classes(img, verbose=5)
        #print(pred)
        if(pred!=label_idx):
            cnt+=1
        count+=1

        #print(cnt)

        
    error= pred/cnt
    print(error, i, cnt, count)

