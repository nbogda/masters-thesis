import keras
from keras.models import load_model
from keras import backend as k
from keras.datasets import mnist
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import random
import sys
import cv2
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import pickle
from filters import filters
import pdb
from mpl_toolkits import mplot3d
import multiprocessing as mp
#to make tensorflow shut up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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


# small helper function for image making
def build_string(param, parameter_names):

	string = ""
	for i in range(0, len(param)):
		string += "%s = %d " % (parameter_names[i], param[i])
	return string


def test_imgs_per_class(model, imgs, labels):
	''' 
	model: keras Sequential model object
		   model with which to make predictions
	imgs: numpy array
		  vector of images to test
	labels: numpy array
			one-hot labels of images, should have same length as imgs
	'''
	
	accuracy = 0
	accuracy_per_class = [0] * 10

	for i in range(0, len(labels)):
		x = imgs[i]/255
		#check to see if image is 4 dimensional
		if len(x.shape) != 4:
			#if not, wrap it in one dimension
			x = np.expand_dims(x, axis=0)
		#get prediction of image
		prediction = model.predict(x)
		#get class label
		pred = np.argmax(prediction[0])
		#get truth
		actual = np.argmax(labels[i])
		#check for accuracy
		if pred == actual:
			accuracy_per_class[pred] += 1
			accuracy += 1
	return accuracy/len(labels), accuracy_per_class


#function to nicely display images in matplotlib
def display_image(model, img, filtered_img, label, param, technique, no, pname, og=False, e=None):
	'''
	model : keras Sequential model object
			the neural network trained on the CINIC 10 images
	x : 224 by 224 by 3 numpy array
		the unaltered, original image
	x_adv : 224 by 224 by 3 numpy array
			the adversarial image
	label : numpy array
			one-hot encoded vector, true labels of the images
	e : int or float
		how much the image was altered, used here to output stats
	'''
	#actual labels of the images
	class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

	#initialize plot
	f = plt.figure(figsize=(30,10))
	plt.title("Class %s example, %s Filtered" % (class_labels[np.argmax(label)], technique))
	plt.axis("off")

	if og:
		og = img
		f_og = filtered_img
		#add og img
		f.add_subplot(1, 2, 1)
		plt.title("Original Image")
		p, c = prediction(model, og)
		og = np.squeeze(og)
		plt.imshow(og)
		plt.annotate("Original Image\nClass prediction: %s\nProbability: %.4f" % (class_labels[p], c), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

		f.add_subplot(1, 2, 2)
		text = build_string(param, pname)
		plt.title("Original %s filtered Image, %s" % (technique, text))
		p, c = prediction(model, f_og)
		f_og = np.squeeze(f_og)
		f_og = np.clip(f_og, 0, 255)
		f_og = f_og / 255
		plt.imshow(f_og)
		plt.annotate("Original Image\nClass prediction: %s\nProbability: %.4f" % (class_labels[p], c), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
		if not os.path.exists("whitebox/clean/%s" % technique):
			os.mkdir("whitebox/clean/%s" % technique)
		plt.savefig("whitebox/clean/%s/og_img_%s_%d_%d.png" % (technique, technique, param[0], no), bbox_inches="tight")
	
	else:
		x = img
		x_adv = filtered_img
		#add adv image
		f.add_subplot(1, 2, 1)
		plt.title("Adversarial Image, e=%d" % e)
		#prediction and confidence
		p, c = prediction(model, x)
		#some preprocessing before sending image off to matplotlib
		x = np.squeeze(x)
		#needed to reverse channels from RGB to BGR so the colors didnt come out blue in matplotlib
		x = np.clip(x, 0, 255)
		x = x / 255
		plt.imshow(x)
		#show annotation with prediction and confidence score
		plt.annotate("Adversarial Image\nClass prediction: %s\nProbability: %.4f" % (class_labels[p], c), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
		#add adversarial image
		f.add_subplot(1, 2, 2)
		text = build_string(param, pname)
		plt.title("%s Filtered Adversarial Image, %s" % (technique, text))
		p, c = prediction(model, x_adv)
		#some preprocessing before sending image off to matplotlib
		x_adv = np.clip(x_adv, 0, 255)
		x_adv = np.squeeze(x_adv)
		x_adv = x_adv / 255
		plt.imshow(x_adv)
		#show annotation with prediction and confidence score
		plt.annotate("%s Filtered Image\nClass prediction: %s\nProbability: %.4f" % (technique, class_labels[p], c), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
		if not os.path.exists("whitebox/clean/%s" % technique):
			os.mkdir("whitebox/clean/%s" % technique)
		plt.savefig("whitebox/clean/%s/adv_img_%s_%d_%d_%d.png" % (technique, technique, param[0], e, no), bbox_inches="tight")


#helper function to predict image class in img_test function
def prediction(model, img):
	'''
	model : keras Sequential model 
			neural network object to prediction with
	img : image of 224 by 224 by 3
		the image to predict
	'''
	#reshape image for keras
	if (len(img.shape) < 4):
		img = np.expand_dims(img, axis=0)
	#make prediction
	pred = model.predict(img)
	#get index of class label
	p = np.argmax(pred[0])
	#get confidence score
	c = max(pred[0])
	#return prediction label and confidence score
	return p, c


def filter_imgs(imgs, method, param_list):
	pool = mp.Pool(int(mp.cpu_count()/5))
	filtered_imgs = pool.starmap(filter_imgs_, [(imgs[i], method, param_list) for i in range(0, len(imgs))])
	pool.close()
	return filtered_imgs


def filter_imgs_(img, method, param_list):
	i = np.squeeze(img)
	filtered_img = None
	if method == "ROF":
		filtered_img = filters.ROF(i, param_list[0])
	elif method == "Nonlocal means":
		filtered_img = filters.nonlocal_means(i, param_list[0], param_list[1], 10, 5, fast=True)
	elif method == "Med Gauss":
		# param 2 should be 1.5
		filtered_img = filters.median(i, param_list[0])
		filtered_img = filters.gaussian(filtered_img, param_list[1], param_list[2])
	elif method == "An Diff":
		filtered_img = filters.anisotropic_diffusion(i, 10, param_list[0], 0.25, param_list[1])
	elif method == "TVD":
		filtered_img = filters.TVD(i, param_list[0])
	return filtered_img


def test_def(model, imgs, labels, method, p, n):

	before_overall_array = []
	before_per_class_array = []
	after_overall_array = []
	after_per_class_array = []

	print("***************NEW FILTER************************")
	filtered_adv = filter_imgs(imgs, method, p)
	print("Applied %s method with param %s, at e=0" % (method, str(p)))
	original_overall, _ = test_imgs_per_class(model, filtered_adv, labels)

	before_overall_array.append(original_overall)
	after_overall_array.append(original_overall)
	
	print("All accuracy: %s" % (original_overall))
	display_image(model, imgs[1], filtered_adv[1], labels[1], p, method, np.random.randint(0, 100000000), n, og=True)

	for i in range(1, 11):
			adv = np.squeeze(np.load("FGSM_imgs/og_img_filter_network/%s/adversarial_images_e=%d.npy" % (method, i)))
			print("\nLoaded in adversaries with e=%d" % i)
			filtered_adv = filter_imgs(adv, method, p)
			print("Applied %s method with param %s" % (method, str(p)))
			
			before_overall, before_per_class = test_imgs_per_class(model, adv, labels)
			after_overall, after_per_class = test_imgs_per_class(model, filtered_adv, labels)
			print("Tested accuracy of adversaries")
			
			print("Before - Accuracy per class: %s" % str(before_per_class))
			print("Before - All accuracy: %.4f" % (before_overall))
			print("After - Accuracy per class: %s" % str(after_per_class))
			print("After - All accuracy: %.4f" % (after_overall))

			'''
			before_overall_array.append(before_overall)
			before_per_class_array.append(before_per_class)
			after_overall_array.append(after_overall)
			after_per_class_array.append(after_per_class)

			# display_image(model, adv[1], filtered_adv[1], labels[1], p, method, np.random.randint(0, 100000000), n, e=i)

	np.save("whitebox/clean/before_overall_%s.npy" % method, before_overall_array)
	np.save("whitebox/clean_per_class/before_per_class_%s.npy" % method, before_per_class_array)
	np.save("whitebox/clean/after_overall_%s.npy" % method, after_overall_array)
	np.save("whitebox/clean_per_class/after_per_class_%s.npy" % method, after_per_class_array)
	'''
def subsample(n, labels):
	
	#ensure even dist from every class
	dict = {}
	for i in range(0, len(labels)):
		t = labels[i]
		if t not in dict:
			dict[t] = []
		dict[t].append(i)

	#return indices across certain range
	indices = []
	for v in dict.values():
		#divide by 10 because we have 10 classes
		sub_indices = list(np.random.choice(v, size=(int(n/10)), replace=False))
		indices += sub_indices
	return indices

global_indices = None

if __name__ == "__main__":

	model_list = ["An Diff", "Med Gauss", "Nonlocal means", "TVD", "ROF"]
	parameters = [[400, 2], [9, 1.5, 5], [10, 60], [130], [65]]
	parameter_names = [["kappa", "option"], ["size", "sigma", "order"], ["sigma", "hval"], ["weight"], ["weight"]]

	imgs = np.load("data/subsample_test_set.npy")
	labels = np.load("data/subsample_test_labels.npy")
	labels = np_utils.to_categorical(labels, 10)
	print("Loaded in %d images" % len(imgs))
	print("Loaded in %d labels" % len(labels))

	for i in range(0, len(model_list)):
		model = load_model("models/%s_vgg16_10_HNN.h5" % model_list[i])
		print("Loaded in %s model" % model_list[i])
		test_def(model, imgs, labels, model_list[i], parameters[i], parameter_names[i])

	#grid_search(imgs, labels, method)	
