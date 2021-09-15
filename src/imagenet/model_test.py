from keras.models import load_model
from keras import backend as k
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import multiprocessing as mp
#to make tensorflow shut up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_imgs(model, imgs, labels, vgg16=False):
    ''' 
    model: keras Sequential model object
           model with which to make predictions
    imgs: numpy array
          vector of images to test
    labels: numpy array
            one-hot labels of images, should have same length as imgs
    '''
    accuracy = 0 
    avg_confidence = 0 

    for i in range(0, len(labels)):
        x = imgs[i]
        #check to see if image is 4 dimensional
        if len(x.shape) != 4:
            #if not, wrap it in one dimension
            x = np.expand_dims(x, axis=0)
        
		#this is for VGG16 network only
        if vgg16:
        	x = preprocess_input(x)
        
		#get prediction of image
        prediction = model.predict(x)
        #get class label
        pred = np.argmax(prediction[0])
        #get truth
        actual = np.argmax(labels[i])
        #get confidence of prediction
        confidence = max(prediction[0])
        avg_confidence += confidence
        #check for accuracy
        if pred == actual:
            accuracy += 1
    return accuracy/len(labels), avg_confidence/len(labels)

if __name__ == "__main__":

	model = load_model("models/vgg16_10_HNN.h5")
	print("Loaded in model")
	#load in saved images
	imgs = np.load("data/subsample_test_set.npy") 
	labels = np.load("data/subsample_test_labels.npy") 
	labels = np_utils.to_categorical(labels, 10)
	print(test_imgs(model, imgs, labels))
	

