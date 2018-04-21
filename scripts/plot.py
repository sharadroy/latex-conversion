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
from keras.utils import plot_model
plot_model(loaded_model, to_file='model.png')