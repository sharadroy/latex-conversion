import cv2
import sys
import numpy as np 
from matplotlib import pyplot as plt 
import os
import tensorflow
import keras
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
import collections
#np.set_printoptions(threshold=np.inf)

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

def create_vectors():
	path= 'extracted_images/'
	files=os.listdir(path)
	print(files)
	cnt=0
	length=16000
	label=np.zeros((length,))
	input_data=np.zeros((length,2025))
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
			ret,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
			#print(count,i,j)
			#hog_feature= find_hog(img)
			img=img.flatten()
			input_data[count]=img
			label[count]=label_idx
			count+=1
			
	np.save('inputs_SVM.npy',input_data)
	np.save('label_SVM.npy',label)
	print(label.shape,input_data.shape)


#create_vectors()
# img=cv2.imread('4.jpg',0)
# #ret,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# hog_feature= find_hog(img)
# plt.imshow(img,cmap='gray')
# plt.show()
# print(hog_feature)

input_data=np.load('inputs_SVM.npy')
label=np.load('label_SVM.npy')
#print(np.isnan(label).any(),input_data)

# train_data= input_data[0:int(0.8*len(label)),:]
# train_label=label[0:int(0.8*len(label))]
# test_data=input_data[int(0.8*len(label)):len(label),:]
# test_label=label[int(0.8*len(label)):len(label)]
# rbf_feature = RBFSampler(gamma=1, random_state=1)
# input_data = rbf_feature.fit_transform(input_data)
train_data, test_data, train_label,test_label = train_test_split(input_data, label, test_size=0.2, shuffle= True, random_state=42)
#print(collections.Counter(train_label),34.0==34)
#clf = svm.NuSVC()
clf = SGDClassifier() 
print("Training in Progress")
clf.fit(train_data,train_label)
print("Training Done")
print("Now Testing")
predicted=clf.predict(test_data)
#print(feature,label)
cnt=0;
for i in range(len(test_label)):
	print(predicted[i],test_label[i])
	if(predicted[i]==test_label[i]):
		cnt+=1
print(cnt)

print(accuracy_score(test_label,predicted))

# final_label=keras.utils.to_categorical(label,num_classes=40)
# print(label.shape,final_label[8500])







