import cv2
import sys
import numpy as np 
from matplotlib import pyplot as plt 
import os
import tensorflow
import keras

def create_data():
	path= 'extracted_images/'
	files=os.listdir(path)
	print(files)
	cnt=0
	length=16000
	label=np.zeros((length,))
	input_data=np.zeros((length,1,45,45))
	data_values=['-', '1','2','+','X', '(',')','=','3','A','N','y','sqrt','4','0','b','z','d','l','sin','5','C','9','6','8','7',
	'times','k','cos','T','alpha','e','int','theta',',','f','infty','sum','tan','pi']
	count=0
	for label_idx, i in enumerate(data_values):
		file_path=os.path.join(path,i)
		new_files=os.listdir(file_path)
		for idx, j in enumerate(new_files):
			if(idx==400):
				break
			img=cv2.imread(os.path.join(file_path,j),0)
			label[count]=label_idx
			input_data[count][0]=img
			count+=1
			print(count,i,j)
			
	np.save('inputs.npy',input_data)
	np.save('label.npy',label)
	print(label.shape,input_data.shape)


#create_data()

input_data=np.load('inputs.npy')
label=np.load('label.npy')
final_label=keras.utils.to_categorical(label,num_classes=40)
print(label.shape,final_label[8500])