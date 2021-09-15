import keras
from keras.models import load_model
from keras.models import model_from_json
from keras import backend as k
from keras.datasets import mnist
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import copy
import sys 
import sklearn.metrics as metrics
import cv2 
import random 
import sys
#to make tensorflow shut up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#function to make adversarial images using fast gradient sign method
def loss_gradient(y_test, x_test, model, epsilon):
	#get shape
	y_true = keras.Input(shape=y_test[0].shape)
	#define loss function
	ce = k.mean(k.categorical_crossentropy(y_true, model.output))
	#get gradients of loss wrt input
	grad_ce = k.gradients(ce, model.inputs)
	#make into function that will get all gradients
	gradients = []
	func = k.function(model.inputs + [y_true], grad_ce)
	#get gradients
	start = 0
	end = int(len(x_test)/5)
	increment = int(len(x_test)/5)
	#doing this in a loop to avoid OOM error
	for i in range(0, 5):
		gradient = func([x_test[start:end], y_test[start:end]])[0] 
		start = end
		end += increment
		gradients.append(gradient)
	gradients = np.array(gradients)
	gradients = gradients.reshape((len(x_test), 32, 32, 3))
	#get gradients sign multiplies by epsilon
	gradients = np.sign(gradients) * epsilon
	#printing stats
	print("Non-zero gradients: %d" % count_nonzero(gradients))
	print("Total gradients: %d" % len(gradients))
	#create images by adding gradients to test images
	adv_imgs = gradients + x_test
	#return adversarial images
	return adv_imgs

#counting the number of gradients that are not completely zero
def count_nonzero(gradients):
	counter = 0
	for i in range(0, len(gradients)):
		if np.any(gradients[i]):
			counter += 1
	return counter


#****************************code to test imgs to see if params need to be tweaked*********************************

#quick function to check if the images are actually working
def img_test(index, adv_imgs, y_labels, x_imgs, model):


	#actual labels of the images
	img_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

	#initialize plot
	f = plt.figure()
	#label plot with truth class label
	y = list(y_labels[index])
	plt.title("Class %s Example" % img_labels[y.index(max(y))])
	plt.axis('off')

	#add subplot
	f.add_subplot(1,2,1)
	plt.title("Original Image")
	#choose an image to test
	og = x_imgs[index]
	#get prediction
	p1, c1 = prediction(model, og) 
	#reshape image for plot
	og = og.reshape((32,32,3))
	plt.imshow(og)
#add annotation with prediction and confidence scores
	plt.annotate("Original Image\nClass prediction: %s\nProbability: %.4f" % (img_labels[p1], c1), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

	#make second subplot
	f.add_subplot(1,2,2)
	#select image
	img = adv_imgs[index]
	#make prediction
	p2, c2 = prediction(model, img)
	plt.title("Adversarial Image")
	#reshape image for plot
	img = img.reshape((32,32,3))
	plt.imshow(img)
	#add annotation with prediction and confidence scores
	plt.annotate("Adversarial Image\nClass prediction: %s\nProbability: %.4f" % (img_labels[p2], c2), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	plt.show()
	fig.savefig("Image_%d.png" % index, bbox_inches="tight")

#helper function to predict image class in img_test function
def prediction(model, image):

	#reshape image for keras
	img = image.reshape(1, 32, 32, 3)
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

#test overall performance
def eval(model, labels, adv_imgs):
	accuracy = 0.0
	avg_confidence = 0.0
	#make predictions on entire test set
	predictions = model.predict(adv_imgs, verbose = 1)
	correct = [] #store correct predictions
	incorrect = [] #store incorrect predictions
	#check accuracy of predictions
	labels = labels
	for i, (l, p) in enumerate(zip(labels, predictions)):
		l = list(l)
		p = list(p)
		predict = p.index(max(p))
		actual = l.index(max(l))
		confidence = max(p)
		#print("Image #%d" % i)
		#print("Predicted class: %d" % p.index(max(p)))
		#print("Actual class: % d" % l.index(max(l)))
		#print("Confidence: %2f" % max(p))
		#print("********************\n")
		if predict == actual:
			accuracy += 1
			correct.append(i)
		else:
			incorrect.append(i)
		avg_confidence += confidence

	accuracy /= len(labels)
	avg_confidence /= len(labels)
	print("\nACCURACY: %2f" % accuracy)
	print("AVERAGE CONFIDENCE: %2f" % avg_confidence)
	print("\nCONFUSION MATRIX:")
	matrix = metrics.confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1))
	print(matrix)
	return accuracy, avg_confidence

#*******************************************************************************************************

if __name__ == "__main__":
	#load model
	print("loading model...")
	model = load_model("model/cifar10_resnet.h5")
		
	print("loading data...")
	#load test data
	x_test = np.load('usable_data/x_test_pixel_mean.npy')
	y_test = np.load('usable_data/y_test.npy')

	#EPSILON
	#e = int(sys.argv[1])
	#epsilon = e/255
	#for making images and plotting info about them
	accuracies = []
	avg_confidences = []
	adv_imgs = None
	print("making images....")
	'''
	for e in range(0, 10):
		adv_imgs = loss_gradient(y_test, x_test, model, e/255)
	
		np.save("adv_imgs/adversarial_images_e=%d.npy" % e, adv_imgs)
		#evaluate model performance on adversarial images
		#return indices of correctly and incorrectly predicted images
		accuracy, avg_confidence = eval(model, y_test, adv_imgs)
		accuracies.append(accuracy)
		avg_confidences.append(avg_confidence)
	'''

	'''
	fig, ax1 = plt.subplots()
	x = np.arange(0, len(accuracies))
	plt.title("Performance of Classifier on Unfiltered Adversarial Images")
	ax1.set_xlabel('Epsilon')
	ax1.set_ylabel('Accuracy')
	ax1.plot(x, accuracies)

	ax2 = ax1.twinx()
	ax2.set_ylabel('Average Confidence')
	ax2.plot(x, avg_confidences, '--', color="gray")
	fig.tight_layout()
	plt.show()
	fig.savefig("Performance.png", bbox_inches="tight")
	'''

	#examine a random image chosen by its index in test set
	#c = random.randint(0, len(incorrect) + 1)
	#getting the location of one of the incorrect predictions
	index = 10 #or 10
	#examine this image that was chosen
	e = 5
	adv_imgs = loss_gradient(y_test, x_test, model, e/255)
	img_test(10, adv_imgs, y_test, x_test, model)
	
