import keras
from keras.models import load_model
from keras import backend as k
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import sys 
import sklearn.metrics as metrics
import cv2 
import sys 
import re
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import csv
#to make tensorflow shut up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#targeted fast gradient sign method, which pushes a class towards a specific label
def single_targeted_FGSM(model, img, label, epsilon):
	'''
	model : keras Sequential model 
			neural network object to get info from
	imgs : images of 224 by 224 by 3
		the input images that we are going to alter
	targets : array 
			one-hot encoding vectors that contains the targeted labels of the classes
	'''
	target = get_target_label(label)
	y_target = keras.Input(shape=target.shape)
	#define loss function
	loss = k.mean(k.categorical_crossentropy(y_target, model.output))
	#get derivative/gradient of loss wrt input
	loss_gradient = k.gradients(loss, model.inputs)
	#build function
	func = k.function(model.inputs + [y_target], loss_gradient)
	#get gradient
	gradient = func([img, target])
	gradient = np.sign(gradient)
	print(gradient)
	gradient = gradient * epsilon
	print(gradient)
	adv_img = img - gradient
	return adv_img


def test_imgs(model, imgs, labels):
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

#function to nicely display images in matplotlib
def display_image(model, x, x_adv, label, e):
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
	f = plt.figure()
	plt.title("Class %s example" % class_labels[np.argmax(label)])
	plt.axis("off")

	#add original image
	f.add_subplot(1, 2, 1)
	plt.title("Original Image")
	#prediction and confidence
	p, c = prediction(model, x)
	#some preprocessing before sending image off to matplotlib
	x = np.squeeze(x)
	#needed to reverse channels from RGB to BGR so the colors didnt come out blue in matplotlib
	plt.imshow(x[...,::-1]) 
	#show annotation with prediction and confidence score
	plt.annotate("Original Image\nClass prediction: %s\nProbability: %.4f" % (class_labels[p], c), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	
	#add adversarial image
	f.add_subplot(1, 2, 2)
	plt.title("Adversarial Image, e=%d" % e)
	p, c = prediction(model, x_adv)
	#some preprocessing before sending image off to matplotlib
	x_adv = np.clip(x_adv, 0, 255)
	x_adv = np.squeeze(x_adv)
	x_adv /= 255
	plt.imshow(x_adv[...,::-1])
	#show annotation with prediction and confidence score
	plt.annotate("Adversarial Image\nClass prediction: %s\nProbability: %.4f" % (class_labels[p], c), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	plt.savefig("results/img_%d.png" % e, bbox_inches="tight")

	
#helper function to predict image class in img_test function
def prediction(model, image):
	'''
	model : keras Sequential model 
			neural network object to prediction with
	img : image of 224 by 224 by 3
		the image to predict
	'''
    #reshape image for keras
	img = image.reshape(1, 224, 224, 3)
    #make prediction
	pred = model.predict(img)
	#flatten
	pred = list(pred[0])
	#get index of class label
	p = pred.index(max(pred))
	#get confidence score
	c = max(pred)
	#return prediction label and confidence score
	return p, c

#regular fast gradient sign method
def FGSM(model, imgs, labels):
	'''
	model : keras Sequential model 
			neural network object to get info from
	imgs : image of 224 by 224 by 3
		the input image that we are going to alter
	labels : array 
			one-hot encoding vector that contains the true label of the class
	'''
	#get shape of label to use later
	y_true = keras.Input(shape=labels[0].shape)
	#define loss function
	loss = k.mean(k.categorical_crossentropy(y_true, model.output))
	#get derivative/gradient of loss wrt input
	loss_gradient = k.gradients(loss, model.inputs)
	#build function
	func = k.function(model.inputs + [y_true], loss_gradient)
	#get gradients
	gradients = []
	start = 0
	end = int(len(labels)/20) 
	increment = int(len(labels)/20) 
	#loop to avoid OOM error
	for i in range(0, 20):
		tmp = func([np.squeeze(imgs[start:end]), labels[start:end]])[0]
		start = end
		end += increment
		gradients.append(tmp)
	gradients = np.array(gradients)
	gradients = np.reshape(gradients, (len(labels), 224, 224, 3))
	return gradients


#targeted fast gradient sign method, which pushes a class towards a specific label
def targeted_FGSM(model, imgs, targets):
	'''
	model : keras Sequential model 
			neural network object to get info from
	imgs : images of 224 by 224 by 3
		the input images that we are going to alter
	targets : array 
			one-hot encoding vectors that contains the targeted labels of the classes
	'''
	y_target = keras.Input(shape=targets[0].shape)
	#define loss function
	loss = k.mean(k.categorical_crossentropy(y_target, model.output))
	#get derivative/gradient of loss wrt input
	loss_gradient = k.gradients(loss, model.inputs)
	#build function
	func = k.function(model.inputs + [y_target], loss_gradient)
	#get gradients
	gradients = []
	start = 0
	end = int(len(targets)/123) 
	increment = int(len(targets)/123) 
	#loop to avoid OOM error
	for i in range(0, 123):
		tmp = func([np.squeeze(imgs[start:end]), targets[start:end]])[0]
		start = end
		end += increment
		gradients.append(tmp)
	gradients = np.array(gradients)
	print(gradients.shape)
	gradients = np.reshape(gradients, (len(targets), 224, 224, 3))
	return gradients

#helper method to make a target label
def get_target_label(label): 
	'''
	label : one-hot vector
			true label of an image
	'''
	true_class = np.argmax(label)
	#push index up by three labels, wrap around in case out of bounds
	target_class = (true_class + 3) % 10
	#convert to one hot vector
	one_hot = np.zeros(10)
	one_hot[target_class] = 1
	return one_hot

def add_imgs(imgs, gradients, epsilon):
	'''
	imgs : numpy array
		   array of original images
	gradients : numpy array
			   array of gradients obtained from fgsm or targeted_fgsm
	epsilon : integer or float
			  number by which to alter the pixel intensities
	'''
	#initialize empty numpy array
	adv_imgs = []
	#create adversarial images here
	for i in range(0, len(imgs)):
		adv = imgs[i] + (np.sign(gradients[i]) * epsilon)
		adv = np.clip(adv, 0, 255)
		adv_imgs.append(adv)
	return adv_imgs

# don't run this again
def fix_channels(name):
	imgs = np.load("data/%s_test_set.npy" % name)
	for i in range(0, len(imgs)):
		imgs[i] = np.clip(imgs[i], 0, 255)
	np.save("data/%s_test_set.npy" % name, imgs)

if __name__ == "__main__":

	model_list = ["subsample", "An Diff", "Med Gauss", "Nonlocal means", "ROF", "TVD"]
	
	# imgs = np.load("data/%d_test_set.npy" % model_list[i])
	imgs = np.load("data/subsample_test_set.npy")
	print("Loaded in %d images" % len(imgs))
	t_labels = np.load("data/subsample_test_labels.npy")
	# print("Loaded in %d labels" % len(t_labels))
	print("Loaded in %d labels" % len(t_labels))
	labels = np_utils.to_categorical(t_labels, 10)
	
	base_test = True
	
	if not base_test:
		for i in range(1, len(model_list)):
			model = load_model("models/%s_vgg16_10_HNN.h5" % model_list[i])
			print("Loaded in %s model" % model_list[i])	

			gradients = None
			if not os.path.exists("data/gradients_HNN_%s_filtered_BUG.npy" % model_list[i]):
				print("Did not find %s gradients, creating them now" % model_list[i])
				gradients = FGSM(model, imgs, labels)
				np.save("data/gradients_HNN_%s_filtered_BUG.npy" % model_list[i], gradients)
				print("Made %s gradients" % model_list[i])
			else:
				print("Loaded in %s gradients" % model_list[i])
				gradients = np.load("data/gradients_HNN_%s_filtered.npy" % model_list[i])
		
			for epsilon in range(1, 11):
				print("Adding e = %d" % epsilon)
				x_adv = add_imgs(imgs, gradients, epsilon)
				if not os.path.exists("FGSM_imgs/og_img_filter_network/%s_BUG" % model_list[i]):
					os.mkdir("FGSM_imgs/og_img_filter_network/%s_BUG" % model_list[i])
				np.save("FGSM_imgs/og_img_filter_network/%s_BUG/adversarial_images_e=%d.npy" % (model_list[i], epsilon), x_adv)
				print("Testing e = %d" % epsilon)
				acc, conf = test_imgs(model, x_adv, labels)
				print("Accuracy: %.4f" % acc)
				print("Confidence: %.4f" % conf)
				del x_adv
			del gradients
			
			print("Completed %s" % model_list[i])
			sys.exit(0)
	
	else:
		for i in range(0, len(model_list)):
			acc = []
			method = model_list[i]
			model = load_model("models/%s_vgg16_10_HNN.h5" % method)
			print(method)
			t_acc, conf = test_imgs(model, imgs, labels)
			print(t_acc)
			acc.append(t_acc)
			for epsilon in range(1, 11):
					print("Testing e = %d" % epsilon)
					adv = np.squeeze(np.load("FGSM_imgs/og_img_filter_network/%s/adversarial_images_e=%d.npy" % (method, epsilon)))
					#if (i == 10):
					#	plt.imshow(adv[1])
					t_acc, conf = test_imgs(model, adv, labels)
					print(t_acc)
					acc.append(t_acc)
		np.save("whitebox/clean/before_overall_%s.npy" % method, acc)
		sys.exit(0)
			
