import keras
from keras.models import load_model
from keras import backend as k
from keras.datasets import mnist
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#to make tensorflow shut up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def eval(adv_imgs, labels, model):

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
		print("Predicted class: %d" % p.index(max(p)))
		print("Actual class: % d" % l.index(max(l)))
		print("Confidence: %2f" % max(p))
		if predict == actual:
			accuracy += 1
		avg_confidence += confidence

	print("Accuracy: %2f" % (accuracy/len(labels))) 
	print("Average Confidence: %2f" % (avg_confidence/len(labels))) 
	return p_labels, c_labels

def view_imgs(adv_imgs, original, p_labels, labels, c_labels, count=5):

	l = list(labels[0])

	f = plt.figure()
	f.add_subplot(1,2,1)
	plt.title("Original Image")
	img = original[0].reshape((28, 28))
	plt.annotate("Class prediction: %d" % (l.index(max(l))), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	plt.imshow(img, cmap='gray')

	f.add_subplot(1,2,2)
	adv = adv_imgs[0].reshape((28,28))
	plt.title("Adversarial Image")
	plt.annotate("Class prediction: %d\nProbability: %.4f" % (p_labels[0], c_labels[0]), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

	plt.imshow(adv, cmap='gray')
	plt.show()



if __name__ == "__main__":

	model = load_model('model/mnist_model.h5')
	adv_imgs = np.load('adv_imgs/adv_imgs.npy')
	original = np.load('adv_imgs/original_imgs.npy')
	labels = np.load('adv_imgs/labels.npy')

	p_labels, c_labels = eval(adv_imgs, labels, model)
	view_imgs(adv_imgs, original, p_labels, labels, c_labels)
