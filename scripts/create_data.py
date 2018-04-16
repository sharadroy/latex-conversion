import cv2
import sys
import numpy as np 
from matplotlib import pyplot as plt 
import os
import tensorflow
import keras
from deskew import deskew 

def create_data():
	path= '..\\extracted_images\\'
	files=os.listdir(path)
	print(files)
	cnt=0
	length=332219
	label=np.zeros((length,))
	input_data=np.zeros((length,45,45))
	data_values=['-', '1','2','+','X', '(',')','=','3','A','N','y','sqrt','4','0','b','z','d','l','sin','5','C','9','6','8','7',
	'times','k','cos','T','alpha','e','int','theta',',','f','infty','sum','tan','pi']
	count=0
	for label_idx, i in enumerate(data_values):
		file_path=os.path.join(path,i)
		new_files=os.listdir(file_path)
		for idx, j in enumerate(new_files):
			# if(idx==2332):
			# 	break
			img=cv2.imread(os.path.join(file_path,j),0)
			# img= deskew(img)
			img= 1*(img>127)
			# img=img.astype('uint8')
			label[count]=label_idx
			input_data[count]=img
			print(input_data.shape,img.shape)
			count+=1
			print(count,i,j)
			
	input_data = input_data.reshape(input_data.shape[0], 1, 45,45)
	input_data = input_data.astype('uint8')
	# input_data /= 255
	print(label.shape,input_data.shape)
	np.save('..\\inputs_4000_binary1.npy',input_data)
	np.save('..\\label_4000_binary1.npy',label)



create_data()

# input_data=np.load('inputs.npy')
# label=np.load('label.npy')
# final_label=keras.utils.to_categorical(label,num_classes=40)
# print(label.shape,final_label[8500])