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
import cv2
#to make tensorflow shut up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#test overall performance
def eval(model, labels, adv_imgs, epsilon):
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
    #class_distribution(predictions, epsilon)
    return accuracy, avg_confidence

def class_distribution(predictions, epsilon):

	img_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship"    , "truck"]

	label_freq = {}
	for y in predictions:
		y = list(y)
		y = y.index(max(y))
		curr = img_labels[y]
		if curr not in label_freq:
			label_freq[curr] = 1
		else:
			label_freq[curr] += 1

	y_pos = np.arange(len(label_freq.keys()))
	plt.figure(figsize=(15, 10))
	plt.bar(y_pos, label_freq.values(), align="center")
	plt.xticks(y_pos, label_freq.keys())
	plt.ylabel("Class Distribution")
	plt.title("Predicted Test Image Class Distribution e=%d" % epsilon)
	plt.savefig("Predict_Dist_e=%d.png" % epsilon)


#for plotting epsilon and accuracy/avg confidence
def plot_results(model, y_test, images):

	accuracies = []
	avg_confidences = []
	for i, n in enumerate(images):
		accuracy, avg_confidence = eval(model, y_test, n, i)
		accuracies.append(accuracy)
		avg_confidences.append(avg_confidence)

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
	fig.savefig("Performance.png", bbox_inches='tight')


def epsilon_comparison(model, y_test, images):

	accuracies = []
	for i in images:
		accuracy, avg_confidence = eval(model, y_test, i, None)
		accuracies.append(accuracy)

	return accuracies	

#editing the images for defense testing
def image_processing_techniques(technique, adversarial, original, model):

	altered_adversarial = []
	altered_original = []

	for a, o in zip(adversarial, original):
		if technique == "Median":
			a = cv2.medianBlur(a, 3)
			o = cv2.medianBlur(o, 3)

		elif technique == "Gaussian":
			o = cv2.GaussianBlur(o, (5,5), 0)
			a = cv2.GaussianBlur(a, (5,5), 0)
		
		elif technique == "Bilateral":
			o = cv2.bilateralFilter(o,3,100,100)
			a = cv2.bilateralFilter(a,3,100,100)

		altered_adversarial.append(a)
		altered_original.append(o)
	
	altered_adversarial = np.array(altered_adversarial)
	altered_original = np.array(altered_original)
	
	return altered_adversarial, altered_original


#examine images
def img_test(adv_imgs, y_labels, x_imgs, model, technique="Unfiltered", epsilon=0):

#actual labels of the images
	img_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

	print(adv_imgs.shape)
	print(x_imgs.shape)

	for index in range(0, 25):

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
		plt.savefig("results/%s/%s_Image_%d, e=%d.png" % (technique, technique, index, epsilon), bbox_inches="tight")

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


if __name__ == "__main__":

	#load model
	print("loading model...")
	model = load_model("model/cifar10_resnet.h5")

	print("loading data...")
	#load original data
	x_test = np.load('usable_data/x_test_pixel_mean.npy')
	y_test = np.load('usable_data/y_test.npy')

	
	images = []
	for i in range(0, 10):
		tmp = np.load("adv_imgs/adversarial_images_e=%d.npy" % i)
		images.append(tmp)
	
	#plot_results(model, y_test, images)
	
	#applying defenses
	print("Applying defenses...")
	adversaries = []
	originals = []
	technique = ["Median", "Bilateral", "Gaussian"]
	for t in technique:
		adv = []
		og = []
		for i in images[9:10]:
			adversarial_images, original_images = image_processing_techniques(t, i, x_test, model)
			adv.append(adversarial_images)
			og.append(original_images)
		adversaries.append(adv)	
		originals.append(og)

	adversaries = np.array(adversaries)
	print(adversaries.shape)
	originals = np.array(originals)

	print("Plotting results.....")
	plt.title("Accuracy vs Epsilon")	
	plt.xlabel("Epsilon")
	plt.ylabel("Accuracy")
	#unfiltered_acc = epsilon_comparison(model, y_test, images)
	#plt.plot(unfiltered_acc, label="Unfiltered")
	for i in range(0, len(technique)):
		#for j in range(0, 10):
		img_test(adversaries[i][0], y_test, originals[i][0], model, technique[i] + "_Filtered_EXAMPLE", 0)
		#filtered_acc = epsilon_comparison(model, y_test, adversaries[i])
		#plt.plot(filtered_acc, label="%s Filtered" % technique[i])
	#plt.legend()
	#plt.savefig("Filtering Results.png", bbox_inches="tight")

