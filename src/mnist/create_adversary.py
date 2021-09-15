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
#to make tensorflow shut up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def eval(model, labels, adv_imgs):

	accuracy = 0.0 
	avg_confidence = 0.0 
	predictions = model.predict(adv_imgs, verbose = 1)
	p_labels = [] #store predicted labels
	c_labels = [] #store confidence scores
	for l, p in zip(labels, predictions):
		l = list(l)
		p = list(p)
		predict = p.index(max(p))
		p_labels.append(predict)
		actual = l.index(max(l))
		confidence = max(p)
		c_labels.append(confidence)
		#print("Image #%d" % i)
		#print("Predicted class: %d" % p.index(max(p)))
		#print("Actual class: % d" % l.index(max(l)))
		#print("Confidence: %2f" % max(p))
		#print("********************\n")
		if predict == actual:
			accuracy += 1
		avg_confidence += confidence

	print("\nACCURACY: %2f" % (accuracy/len(labels))) 
	print("AVERAGE CONFIDENCE: %2f" % (avg_confidence/len(labels)))
	print("\nCONFUSION MATRIX:")
	matrix = metrics.confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1))
	print(matrix)
	return p_labels, c_labels

def loss_gradient_mthd_2(y_test, x_test, model, epsilon):

	y_true = keras.Input(shape=y_test[0].shape)
	#loss function
	ce = k.mean(k.categorical_crossentropy(y_true, model.output))
	#gradient of loss wrt input
	grad_ce = k.gradients(ce, model.inputs)
	#turning it into a function
	func = k.function(model.inputs + [y_true], grad_ce)
	#get sign of gradient * epsilon
	gradients_b = func([x_test, y_test])[0]
	#grabbing some gradients indices corresponding to the class
	indices = []
	#getting substitute gradients
	for i in range(0, 10):
		for j in range(0, len(gradients_b)):
			c = list(y_test[j])
			curr_class = c.index(max(c))
			if np.any(gradients_b[j]) and curr_class == i:
				indices.append(j)
				break
	#print(indices)
	gradients = np.sign(gradients_b) * epsilon
	#count amount of non-zero gradients
	counter = 0
	for i in range(0, len(gradients)):
		if np.any(gradients[i]):
			#counting non-zero gradients
			counter += 1
		else:
			#making zero gradients non-zero by substituting in a gradient from the same class that is non-zero
			c = list(y_test[i])
			curr_class = c.index(max(c))
			gradients[i] = gradients[indices[curr_class]]

	print("Non-zero gradients: %d\nTotal gradients: %d\n" % (counter, len(gradients)))
	print(counter2)
	#create images
	adv_imgs = gradients + x_test
	#save numpy array into file 
	np.save('adv_imgs/adv_imgs.npy', adv_imgs)
	return adv_imgs, gradients_b

#quick function to check if the images are actually working
def img_test(index, adv_imgs, y_labels, x_imgs, model):

	#initialize plot
	f = plt.figure()
	#label plot with truth class label
	y = list(y_labels[index])
	plt.title("Class %d Example" % y.index(max(y)))
	plt.axis('off')

	#add subplot
	f.add_subplot(1,2,1)
	plt.title("Original Image")
	#choose an image to test
	og = x_imgs[index]
	#get prediction
	p1, c1 = prediction(model, og)
	#reshape image for plot
	og = og.reshape((28,28))
	plt.imshow(og, cmap='gray')
	#add annotation with prediction and confidence scores
	plt.annotate("Original Image\nClass prediction: %d\nProbability: %.4f" % (p1, c1), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	
	#make second subplot
	f.add_subplot(1,2,2)
	#select image
	img = adv_imgs[index]
	#make prediction
	p2, c2 = prediction(model, img)
	plt.title("Adversarial Image")
	#reshape image for plot
	img = img.reshape((28,28))
	plt.imshow(img, cmap='gray')
	#add annotation with prediction and confidence scores
	plt.annotate("Adversarial Image\nClass prediction: %d\nProbability: %.4f" % (p2, c2), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	#plt.savefig(("results/example_%d" % index), bbox_inches="tight")
	plt.show()

def prediction(model, image):

	#reshape image for keras
	img = image.reshape(1, 28, 28, 1)
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

def examine_imgs(y_test, x_test, adv_imgs, gradients, p_labels, c_labels, model):

	indices_visited = []
	counter = 0.0
	accurate = 0.0
	confidence = 0.0

	for i in range(len(gradients)):
		#checking for all zero gradients
		if not np.any(gradients[i]):
			#checking to see if confidence score is really high
			#(it is, all 0.9999 and 1.0s)
			'''
			print("Adversarial confidence score: %.4f" % c_labels[i])
			print("Predicted adversarial class: %d" % p_labels[i])
			p, c = prediction(model, x_test[i])
			print("Predicted actual class: %d" % p)
			print("Actual confidence score: %.4f" % c)
			print("Actual class: %d" % y_t.index(max(y_t)))
			print("\n ********************************* \n")
			'''
			y_t = list(y_test[i])
			indices_visited.append(i)
			confidence += c_labels[i]
			if y_t.index(max(y_t)) == p_labels[i]:
				accurate += 1
			
		'''	
		elif c_labels[i] >= 0.9999 and i not in indices_visited:
			
			print(gradients[i])	
			print("Adversarial confidence score: %.4f" % c_labels[i])
			print("Predicted adversarial class: %d" % p_labels[i])
			p, c = prediction(model, x_test[i])
			print("Predicted actual class: %d" % p)
			print("Actual confidence score: %.4f" % c)
			print("Actual class: %d" % y_t.index(max(y_t)))
			print("\n ********************************* \n")
			confidence += c_labels[i]
			y_t = list(y_test[i])
			if y_t.index(max(y_t)) == p_labels[i]:
				accurate += 1
			counter += 1
		'''

	counter = len(indices_visited)
	print("Average confidence: %.4f" % (confidence/counter))
	#print("Accuracy: %.4f" % (accurate/counter))

if __name__ == "__main__":
	#n_model = load_model("model/mnist_resnet.h5")
	n_model = load_model("model/mnist_model.h5")

	#print(n_model.summary())
	#sys.exit()

	#load in data
	y_test = np.load('adv_imgs/y_test.npy')
	x_test = np.load('adv_imgs/x_test.npy')
	
	#x_test = add_noise(x_test)
	#img = x_test[0].reshape((28,28))
	#plt.imshow(img, cmap='gray')
	#plt.show()
		
	#EPSILON	
	epsilon = 40/255
	print("Creating adversarial image.....\n")
	adv_imgs, gradients = loss_gradient_mthd_2(y_test, x_test, n_model, epsilon)

	print("Plotting images....")
	
	#choose a random image to examine from the test images data by its index
	index = 4
	#check overall accuracy
	p_labels, c_labels = eval(n_model, y_test, adv_imgs)
	#look at loss closely
	#examine_imgs(y_test, x_test, adv_imgs, gradients, p_labels, c_labels, n_model)
	#visualze one image in particular by index
	#index = 43
	#img_test(index, adv_imgs, y_test, x_test, n_model)
	

