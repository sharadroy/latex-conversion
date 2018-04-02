import cv2
import sys
import numpy as np 
from matplotlib import pyplot as plt 
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
input_data= np.load('../inputs.npy')
input_data = input_data.reshape(input_data.shape[0],45,45,1)
print(input_data.shape)
labels= np.load('../label.npy')
labels=keras.utils.to_categorical(labels,40)
X_train, X_test, y_train, y_test = train_test_split( input_data, labels, test_size=0.2, shuffle= True, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)

def cnn_model():

	num_classes=40
	model = Sequential()
	model.add(Convolution2D(30, (5,5), input_shape=(45,45,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

model=cnn_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)
