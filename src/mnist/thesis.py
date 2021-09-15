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
from art.attacks.evasion import FastGradientMethod as fgsm
from art.attacks.evasion import ProjectedGradientDescent as pgd
from art.classifiers import KerasClassifier
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2

def img_test(index, adv_imgs, y_labels, x_imgs, model, attack, c):

	#actual labels of the images
	#img_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

	#initialize plot
	f = plt.figure()
	#label plot with truth class label
	y = list(y_labels[index])
	plt.title("%s %s Example" % (attack, y.index(max(y))))
	plt.axis('off')

	#add subplot
	f.add_subplot(1,2,1)
	plt.title("Adversarial Image")
	#choose an image to test
	og = x_imgs[index]
	#get prediction
	p1, c1 = prediction(model, og) 
	#reshape image for plot
	og = og.reshape((28, 28))
	plt.imshow(og, cmap='gray')
	#add annotation with prediction and confidence scores
	plt.annotate("Adversarial Image\nClass prediction: %s\nProbability: %.4f" % (p1, c1), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

	'''
	#make second subplot
	f.add_subplot(1,3,2)
	#select image
	img = adv_imgs[index]
	difference = cv2.subtract(img, og) 
	#difference = difference * (1000/255)
	plt.title("Difference")
	plt.imshow(difference, cmap='gray')
	'''

	img = adv_imgs[c]
	
	#make third subplot
	f.add_subplot(1,2,2)
	p2, c2 = prediction(model, img)
	plt.title("Bilateral Filtered Image")
	#rehape image for plot
	img = img.reshape((28,28))
	plt.imshow(img, cmap='gray')
	#add annotation with prediction and confidence scores
	plt.annotate("Filtered Image\nClass prediction: %s\nProbability: %.4f" % (p2, c2), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	#plt.show()
	plt.savefig('results/%s/adv_%s_example_img_%s.png' % (attack, attack, np.random.randint(1000)), bbox_inches="tight")

#helper function to predict image class in img_test function
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

def accuracy_score(model, imgs, labels):
	pred = model.predict(imgs)
	pred = np.argmax(pred, axis=1)
	accuracy = metrics.accuracy_score(labels, pred)
	return accuracy


def attack(model, imgs, labels):
	attack = "PGD"  # can be FGSM or PGD
	labels = np.argmax(labels, axis=1)
	
	accuracies = []
	advs = []

	accuracy = accuracy_score(model, imgs, labels)
	print(accuracy)
	accuracies.append(accuracy)

	for i in range(1, 50, 5):
		method = None
		if attack == "FGSM":
			method = fgsm(model, eps=(i/255))
		elif attack == "PGD":
			eps = i/255
			method = pgd(model, eps=eps, eps_step=eps*0.5, max_iter=10)

		adv = method.generate(imgs)
		advs.append(adv)
		accuracy = accuracy_score(model, adv, labels)
		print(accuracy)
		accuracies.append(accuracy)
	
	np.save("%s_advs.npy" % attack, advs)
	np.save("results/%s/base_accuracy.npy" % attack, accuracies)

def show_results(model, test, labels):
	attack = "PGD"
	advs = np.load("%s_advs.npy" % attack)
	
	indices = [1, 500, 6000]
	eps = np.arange(1, 50, 5)

	fig = plt.figure(1, figsize=(7, 3)) 
	plt.axis('off')
	grid = ImageGrid(fig, 111, nrows_ncols=(3, len(eps) + 1))

	imgs = []
	for j in indices:
		imgs.append(test[j])
		for i in range(0, len(eps)):
			imgs.append(advs[i][j])
	
	for ax, im in zip(grid, imgs):
		im = im.reshape((28,28))
		ax.imshow(im, cmap='gray')
		ax.axis('off')

	plt.savefig("results/%s/init_adv.png" % attack)

def img_processing_techniques(imgs, technique, eps):

	filtered_imgs = []
	for img in imgs:
		img = img.reshape((28, 28))
		img = img.astype('float32')

		#use technique parameter to determine operation to perform

		#threshold images
		if technique == "Thresholded":
			_, img = cv2.threshold(img, 100/255, 1, cv2.THRESH_BINARY)
		#gaussian blur images
		elif technique == "Gaussian":
			img = cv2.GaussianBlur(img, (7,7), 10)
		#median filter images
		elif technique == "Median":
			img = cv2.medianBlur(img,5)
		#bilateral filter images
		elif technique == "Bilateral":
			img = cv2.bilateralFilter(img,5,100,100)
		#resize images
		elif technique == "Resize":
			img = cv2.resize(img, (7, 7), interpolation = cv2.INTER_AREA)
			img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_LANCZOS4)

		#reshape images back into one dimensional vectors
		filtered_imgs.append(img)

	# show all images in a grid
	filtered_imgs = np.array(filtered_imgs).reshape(len(filtered_imgs), 28, 28, 1)
	
	'''
	if technique != "Thresholded":
		plt.imshow(filtered_imgs[0].reshape((28,28)), cmap='gray')
		plt.title("%s Filtered, e = %d" % (technique, eps))
		plt.show()
	'''
	return filtered_imgs

# load in blackbox model for blackbox comparison too :)
def defend(model, x_test, labels):
	attacks = ["PGD", "FGSM"]
	filtering_methods = ["Resize", "Thresholded", "Gaussian", "Median", "Bilateral"]
	for attack in attacks:
		display_images = []
		advs = np.load("%s_advs.npy" % attack)
		# add original FGSM to front
		all_accuracies = []
		for f in filtering_methods:
			per_filter = []
			print("Testing %s" % f) 
			accuracies = []
			imgs = img_processing_techniques(x_test, f, 0)
			per_filter.append([x_test[0], imgs[0]])
			accuracy = accuracy_score(model, imgs, labels)
			accuracies.append(accuracy)
			print(accuracy)
			for i, a in enumerate(advs):
				imgs = img_processing_techniques(a, f, i)
				per_filter.append([a[0], imgs[0]])
				accuracy = accuracy_score(model, imgs, labels)
				accuracies.append(accuracy)
				print(accuracy)
			all_accuracies.append(accuracies)
			display_images.append(per_filter)
		np.save("results/%s/blackbox_defense_testing.npy" % attack, all_accuracies)
		#np.save("results/%s/filtered_imgs.npy" % attack, display_images)
				
def show_filtered(attack):
	filtering_methods = ["Resize", "Thresholded", "Gaussian", "Median", "Bilateral"]
	
	eps = np.arange(1, 50, 5)

	fig = plt.figure(1, figsize=(5, 8)) 
	plt.axis('off')
	grid = ImageGrid(fig, 111, nrows_ncols=(6, 3))

	final_imgs = []
	flag = False
	imgs = np.load("results/%s/filtered_imgs.npy" % attack)
	for i in imgs:
		low = i[0]
		med = i[4]
		high = i[10]
		if not flag:
			final_imgs.append(low[0])
			final_imgs.append(med[0])
			final_imgs.append(high[0])
			flag = True
		
		final_imgs.append(low[1])
		final_imgs.append(med[1])
		final_imgs.append(high[1])
		
	for ax, im in zip(grid, final_imgs):
		im = im.reshape((28,28))
		ax.imshow(im, cmap='gray')
		ax.axis('off')

	plt.savefig("results/%s/compare_def.png" % attack)


def black_box(x_test, y_test):
	black_model = load_model("model/mnist_model_2.h5")
	attacks = ["FGSM", "PGD"]
	for a in attacks:
		advs = np.load("%s_advs.npy" % a)
		accuracies = []
		accuracy = 	accuracy_score(black_model, x_test, y_test)
		accuracies.append(accuracy)
		print(accuracy)
		for v in advs:
			accuracy = 	accuracy_score(black_model, v, y_test)
			accuracies.append(accuracy)
			print(accuracy)
		#np.save("results/%s/blackbox.npy" % a, accuracies)
	
def subsample(n, labels):
   
    #ensure even dist from every class
    dict = {}
    for i in range(0, len(labels)):
        t = np.argmax(labels[i])
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


def make_grid(x_test, y_test):
    ind = subsample(1000, y_test)
    imgs = x_test[ind]
    fig = plt.figure(1, figsize=(20, 10)) 
    plt.axis('off')
    grid = ImageGrid(fig, 111, nrows_ncols=(20, 50))
    
    for ax, im in zip(grid, imgs):
        im = im.reshape((28,28))
        ax.imshow(im, cmap='gray')
        ax.axis('off')

    #plt.show()
    plt.savefig("img_grid.png")

def wrong_o(advs, model, y_test):
	ind_list = []
	for i, (a, l) in enumerate(zip(advs, y_test)):
		pred = np.argmax(model.predict(a.reshape(1, 28, 28, 1)))
		truth = np.argmax(l)
		if truth != pred:
			ind_list.append(i)
			if len(ind_list) == 100:
				return ind_list


def correct_o(advs, model, y_test):
	ind_list = []
	imgs = []
	for i, (a, l) in enumerate(zip(advs, y_test)):
		truth = np.argmax(l)
		pred = np.argmax(model.predict(a.reshape(1, 28, 28, 1)))
		if truth == pred:
			continue
		a = cv2.bilateralFilter(a,5,100,100)
		pred = np.argmax(model.predict(a.reshape(1, 28, 28, 1)))
		if truth == pred:
			imgs.append(a)
			ind_list.append(i)
			if len(ind_list) == 10:
				return ind_list, imgs


def main():
	model = load_model("model/mnist_model.h5")
	y_test = np.load('adv_imgs/y_test.npy')
	x_test = np.load('adv_imgs/x_test.npy')
	#make_grid(x_test, y_test)
	# black_box(x_test, np.argmax(y_test, axis=1))
	# model = KerasClassifier(model=model)
	# attack(model, x_test, y_test)
	#defend(model, x_test, np.argmax(y_test, axis=1))
	# show_results(None, x_test, np.argmax(y_test, axis=1))
	#show_filtered("FGSM")
	# 40 is a good one
	attacks = ["PGD"]
	for attack in attacks:
		advs = np.load("%s_advs.npy" % attack)[8]
		#ind = wrong_o(advs, model, y_test)
		ind, imgs = correct_o(advs, model, y_test)
		for i, c in enumerate(ind):
			img_test(c, imgs, y_test, advs, model, attack, i)

if __name__ == "__main__":
	main()
