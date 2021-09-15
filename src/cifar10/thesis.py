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

def img_test(index, adv_imgs, y_labels, x_imgs, model, attack, c, filter):

	#actual labels of the images
	img_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

	#initialize plot
	f = plt.figure()
	#label plot with truth class label
	y = list(y_labels[index])
	plt.title("%s %s Example" % (attack, img_labels[y.index(max(y))]))
	plt.axis('off')

	#add subplot
	f.add_subplot(1,2,1)
	plt.title("Adversarial Image")
	#choose an image to test
	og = x_imgs[index]
	#get prediction
	p1, c1 = prediction(model, og) 
	#reshape image for plot
	og = og.reshape((32,32,3))
	og = np.clip(og + (50/255), 0, 1)
	plt.imshow(og)
	#add annotation with prediction and confidence scores
	plt.annotate("Adversarial Image\nClass prediction: %s\nProbability: %.4f" % (img_labels[p1], c1), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

	'''
	#make second subplot
	f.add_subplot(1,3,2)
	#select image
	img = adv_imgs[index]
	difference = cv2.subtract(img, og)
	difference = difference * 150
	difference = difference.astype(np.uint8)
	plt.title("Difference")
	plt.imshow(difference)
	'''

	img = adv_imgs[c]
	#make third subplot
	f.add_subplot(1,2,2)
	p2, c2 = prediction(model, img)
	plt.title("%s Filtered Image" % filter)
	#reshape image for plot
	img = img.reshape((32,32,3))
	img = np.clip(img + (50/255), 0, 1)
	plt.imshow(img)
	#add annotation with prediction and confidence scores
	plt.annotate("Filtered Image\nClass prediction: %s\nProbability: %.4f" % (img_labels[p2], c2), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	#plt.show()
	plt.savefig('paper_results/%s/%s_example_img_%d.png' % (attack, attack, np.random.randint(1000)), bbox_inches="tight")
	
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


def accuracy_score(model, imgs, labels):
	pred = model.predict(imgs)
	pred = np.argmax(pred, axis=1)
	accuracy = metrics.accuracy_score(labels, pred)
	return accuracy


def attack(model, imgs, labels, attack):
	labels = np.argmax(labels, axis=1)
	
	accuracies = []
	advs = []

	print("Running %s" % attack)

	print("Step %d" % 0)
	accuracy = accuracy_score(model, imgs, labels)
	print(accuracy)
	accuracies.append(accuracy)

	if attack == "PGD":
		advs = np.load('pgd_imgs/PGD_advs.npy')
	for i in range(1, 10):
		print("Step %d" % i)
		method = None
		adv = None
		if attack == "FGSM":
			#adv = np.load('fgsm_imgs/adversarial_images_e=%d.npy' % i)
			method = fgsm(model, eps=i/255)
			adv = method.generate(imgs)
			advs.append(adv)
		elif attack == "PGD":
			adv = advs[i]	
			#eps = i/255
			#method = pgd(model, eps=eps, eps_step=eps*0.5, max_iter=5)
			#adv = method.generate(imgs)
			#advs.append(adv)
		
		accuracy = accuracy_score(model, adv, labels)
		print(accuracy)
		accuracies.append(accuracy)
		#np.save("%s_adv=%d.npy" % (attack, i), advs)
	
	np.save("%s_advs.npy" % attack, advs)
	np.save("paper_results/%s/base_accuracy_art.npy" % attack, accuracies)

def show_results(model, test, labels):
	attack = "PGD"

	advs = []
	
	advs.append(test)
	
	if attack == "FGSM":
		for i in range(0, 10):
			advs.append(np.load("fgsm_imgs/adversarial_images_e=%d.npy" % i))
	elif attack == "PGD":
		pgd = np.load("pgd_imgs/PGD_advs.npy")
		advs = np.concatenate((advs, pgd), axis=0)

	indices = [1, 1000, 9999]
	eps = np.arange(1, 10)

	fig = plt.figure(1, figsize=(7, 3)) 
	plt.axis('off')
	grid = ImageGrid(fig, 111, nrows_ncols=(3, len(eps) + 1))

	imgs = []
	for j in indices:
		imgs.append(test[j])
		for i in range(0, len(eps)):
			imgs.append(advs[i][j])
	
	brightening = 50
	for ax, im in zip(grid, imgs):
		#im = im.reshape((32,32))
		im = np.clip(im + (brightening/255), 0, 1)
		ax.imshow(im)
		ax.axis('off')

	plt.show()
	#plt.savefig("paper_results/%s/cifar_init_adv.png" % attack)

def img_processing_techniques(imgs, technique, eps):

	filtered_imgs = []
	for img in imgs:
		#img = img.reshape((32, 32))
		#img = img.astype('float32')

		#use technique parameter to determine operation to perform

		#gaussian blur images
		if technique == "Gaussian":
			img = cv2.GaussianBlur(img, (5,5), 1)
		#median filter images
		elif technique == "Median":
			img = cv2.medianBlur(img,5)
		#bilateral filter images
		elif technique == "Bilateral":
			img = cv2.bilateralFilter(img,5,100,100)
		#resize images
		elif technique == "Resize":
			img = cv2.resize(img, (17, 17), interpolation = cv2.INTER_AREA)
			img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_LANCZOS4)

		#reshape images back into one dimensional vectors
		filtered_imgs.append(img)

	# show all images in a grid
	filtered_imgs = np.array(filtered_imgs).reshape(len(filtered_imgs), 32, 32, 3)
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
	filtering_methods = ["Resize", "Gaussian", "Median", "Bilateral"]
	for attack in attacks:
		advs = []
		if attack == "PGD":
			advs = np.load('pgd_imgs/PGD_advs.npy')
		elif attack == "FGSM":
			advs = []
			for i in range(1, 10):
				adv = np.load('fgsm_imgs/adversarial_images_e=%d.npy' % i)
				advs.append(adv)
		display_images = []
		# add originals to front
		all_accuracies = []
		for f in filtering_methods:
			per_filter = []
			print("Testing %s" % f) 
			accuracies = []
			print(0)
			imgs = img_processing_techniques(x_test, f, 0)
			per_filter.append([x_test[0], imgs[0]])
			accuracy = accuracy_score(model, imgs, labels)
			accuracies.append(accuracy)
			print(accuracy)
			for i, a in enumerate(advs):
				print(i + 1)
				imgs = img_processing_techniques(a, f, i)
				per_filter.append([a[0], imgs[0]])
				accuracy = accuracy_score(model, imgs, labels)
				accuracies.append(accuracy)
				print(accuracy)
			all_accuracies.append(accuracies)
			display_images.append(per_filter)
		#np.save("paper_results/%s/cifar10_defense_testing_%s_BLACKBOX.npy" % (attack, attack), all_accuracies)
		#np.save("paper_results/%s/cifar10_filtered_imgs_%s.npy" % (attack, attack), display_images)
				

#subsample class examples with even distribution
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
		im = im + (50/255)
		#im = im[...,::-1]
		ax.imshow(im, cmap='gray')
		ax.axis('off')

	#plt.show()
	plt.savefig("img_grid.png")

def show_filtered(attack):
	filtering_methods = ["Resize", "Gaussian", "Median", "Bilateral"]
	
	eps = np.arange(1, 10)

	fig = plt.figure(1, figsize=(5, 8)) 
	plt.axis('off')
	grid = ImageGrid(fig, 111, nrows_ncols=(6, 3))

	final_imgs = []
	flag = False
	imgs = np.load("paper_results/%s/cifar10_filtered_imgs_%s.npy" % (attack, attack))
	for i in imgs:
		low = i[0]
		med = i[4]
		high = i[9]
		if not flag:
			final_imgs.append(low[0])
			final_imgs.append(med[0])
			final_imgs.append(high[0])
			flag = True
		
		final_imgs.append(low[1])
		final_imgs.append(med[1])
		final_imgs.append(high[1])
		
	for ax, im in zip(grid, final_imgs):
		im = im + (50/255)
		im = im[...,::-1]
		ax.imshow(im, cmap='gray')
		ax.axis('off')

	plt.savefig("paper_results/%s/compare_def_%s.png" % (attack, attack))


def black_box(x_test, y_test):
	black_model = load_model("model/cifar10_model_BLACKBOX.h5")
	attacks = ["FGSM", "PGD"]
	for a in attacks:
		advs = []
		if a == "PGD":
			advs = np.load('pgd_imgs/PGD_advs.npy')
		elif a == "FGSM":
			advs = []
			for i in range(1, 10):
				adv = np.load('fgsm_imgs/adversarial_images_e=%d.npy' % i)
				advs.append(adv)
		accuracies = []
		accuracy = 	accuracy_score(black_model, x_test, y_test)
		accuracies.append(accuracy)
		print(accuracy)
		for v in advs:
			accuracy = 	accuracy_score(black_model, v, y_test)
			accuracies.append(accuracy)
			print(accuracy)
		np.save("paper_results/%s/blackbox_model2.npy" % a, accuracies)
	
def plot_results():
    arr = np.arange(0, 10) 
    #arr = np.array(arr)/255
    #whitebox_fgsm = np.load('paper_results/FGSM/base_accuracy.npy')
    whitebox_pgd = np.load('paper_results/PGD/base_accuracy.npy')
    #blackbox_pgd = np.load('paper_results/PGD/blackbox_resnet.npy')
    #blackbox_fgsm = np.load('paper_results/FGSM/blackbox_resnet.npy')
    #blackbox_pgd_2 = np.load('paper_results/PGD/blackbox_model2.npy')
    #blackbox_fgsm_2 = np.load('paper_results/FGSM/blackbox_model2.npy')
    fig = plt.figure(figsize=(10, 5))
    #plt.plot(arr, whitebox_fgsm, label="Whitebox FGSM")
    plt.plot(arr, whitebox_pgd, label="Whitebox PGD")
    #plt.plot(arr, blackbox_fgsm, label="Blackbox ResNet FGSM")
    #plt.plot(arr, blackbox_pgd, label="Blackbox ResNet PGD")
    #plt.plot(arr, blackbox_fgsm_2, label="Blackbox Network #2 FGSM")
    #plt.plot(arr, blackbox_pgd_2, label="Blackbox Network #2 PGD")
    plt.title("CIFAR-10 White-Box PGD")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('paper_results/cifar10_pgd_only.png')

def compare_def(attack):
	filters = ["Resized", "Gaussian", "Median", "Bilateral"]
	arr = np.arange(0, 10)
	#arr = np.array(arr)/255
	whitebox = np.load('paper_results/%s/base_accuracy.npy' % attack)
	all_accuracies = np.load('paper_results/%s/cifar10_defense_testing_%s.npy' % (attack, attack))
	#blackbox = np.load('paper_results/%s/blackbox_resnet.npy' % attack)
	fig = plt.figure()
	plt.plot(arr, whitebox, '--', label="Base %s" % attack)
	for i in range(0, len(all_accuracies)):
		plt.plot(arr, all_accuracies[i], label=filters[i])
	plt.title("CIFAR-10 Whitebox %s Defenses Comparison" % attack)
	plt.xlabel("Epsilon")
	plt.ylabel("Accuracy")
	plt.legend()
	#plt.show()
	plt.savefig('paper_results/%s/cifar10_wb_def_comparison_%s.png' % (attack, attack))

def wrong_o(advs, model, y_test):
	ind_list = []
	for i in range(0, len(advs), 100):
		a = advs[i]
		l = y_test[i]
		pred = np.argmax(model.predict(a.reshape(1, 32, 32, 3)))
		truth = np.argmax(l)
		if truth != pred:
			ind_list.append(i)
			if len(ind_list) == 30:
				return ind_list


def correct_o(advs, model, y_test):
    ind_list_a = []
    ind_list_b = []
    imgs_a = []
    imgs_b = []
    for i, (a, l) in enumerate(zip(advs, y_test)):
        truth = np.argmax(l)
        pred = np.argmax(model.predict(a.reshape(1, 32, 32, 3)))
        if truth == pred:
            continue
        a = cv2.GaussianBlur(a, (5,5), 1)
        b = cv2.resize(a, (17, 17), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (32, 32), interpolation=cv2.INTER_LANCZOS4)
        pred_a = np.argmax(model.predict(a.reshape(1, 32, 32, 3)))
        pred_b = np.argmax(model.predict(a.reshape(1, 32, 32, 3)))
        if truth == pred_a:
            imgs_a.append(a)
            ind_list_a.append(i)
        if truth == pred_b:
            imgs_b.append(b)
            ind_list_b.append(i)
        if len(ind_list_a) == 30 and len(ind_list_b) == 30: 
            return ind_list_a, ind_list_b, imgs_a, imgs_b

def main():
	model = load_model("model/cifar_resnet.h5")
	y_test = np.load('usable_data/y_test.npy')
	x_test = np.load('usable_data/x_test_pixel_mean.npy')
	#make_grid(x_test, y_test)
	#attacks = ["PGD", "FGSM"]
	#for a in attacks:
	#	compare_def(a)
	#plot_results()
	#black_box(x_test, np.argmax(y_test, axis=1))
	#model = KerasClassifier(model=model)
	#attacks = ['FGSM']
	#for a in attacks:
	#attack(model, x_test, y_test, "PGD")
	#defend(model, x_test, np.argmax(y_test, axis=1))
	#show_results(None, x_test, np.argmax(y_test, axis=1))
	#show_filtered("PGD")
	#plot_results()
	#advs = None
	#attack = "FGSM"
	#if attack == "PGD":
	#	pgd_advs = np.load('pgd_imgs/PGD_advs.npy')
	#	advs = pgd_advs[4]
	#elif attack == "FGSM":
	#	advs = np.load('fgsm_imgs/adversarial_images_e=4.npy')
	
	'''
	model = KerasClassifier(model=model)
	eps = 5/255
	method = pgd(model, eps=eps, eps_step=eps*0.5, max_iter=5)
	advs = method.generate(x_test)
	np.save('pgd_imgs_e=5.npy', advs)
	print("Done making advs")
	'''
	
	advs = np.load('pgd_imgs_e=5.npy')
	#ind = wrong_o(advs, model, y_test)
	ind_a, ind_b, imgs_a, imgs_b = correct_o(advs, model, y_test)
	for i, c in enumerate(ind_a):
		img_test(c, imgs_a, y_test, advs, model, "PGD", i, "Gaussian")
	for i, c in enumerate(ind_b):
		img_test(c, imgs_b, y_test, advs, model, "PGD", i, "Resize")

if __name__ == "__main__":
	main()
