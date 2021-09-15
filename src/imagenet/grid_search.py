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
from filters import filters
import multiprocessing as mp

def accuracy_score(model, imgs, labels):
	tmp = []
	for i in imgs:
		tmp.append(i.astype('float32'))

	pred = model.predict(tmp)
	pred = np.argmax(pred, axis=1)
	accuracy = metrics.accuracy_score(labels, pred)
	return accuracy

def filter_imgs(imgs, method, param_list):
    pool = mp.Pool(int(mp.cpu_count()/2))
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
		filtered_img = filters.anisotropic_diffusion(i, param_list[0], param_list[1])
	elif method == "TVD":
		filtered_img = filters.TVD(i, param_list[0])
	elif method == "Resize":
		filtered_img = cv2.resize(i, (param_list[0], param_list[0]), interpolation = cv2.INTER_AREA)
		filtered_img = cv2.resize(filtered_img, (224, 224), interpolation = cv2.INTER_LANCZOS4)

	return filtered_img

def get_params(method):

	if method == "ROF":
		return [65]
	elif method == "Nonlocal means":
		return [10, 60]
	elif method == "Med Gauss":
		return [5, 20, 11]
	elif method == "An Diff":
		return [0.1, 20]
	elif method == "TVD":
		return [130]
	elif method == "Resize":
		return [30]


#load in blackbox model for blackbox comparison too :)
def defend(model, x_test, labels, attack, filter, blackbox):
	model = KerasClassifier(model=model)
	print("Running %s" % attack)
	advs = []
	advs.append(x_test)
	# add originals to front
	all_accuracies = []
	filtering_methods = [filter]
	for f in filtering_methods:
		print("Testing %s" % f)  
		accuracies = []
		print(0)
		params = get_params(f)
		
		imgs = filter_imgs(x_test, f, params)
		accuracy = accuracy_score(model, imgs, labels)
		accuracies.append(accuracy)
		print(accuracy)
		
		for i in range(1, 10, 2):
			adv = None
			print(i)
			if attack == "PGD":
				adv = np.load('PGD_imgs/filters/%s/adversarial_images_e=%d.npy' % (f, i))
			elif attack == "FGSM": 
				try:
					adv = np.load('FGSM_imgs/filter_double/%s/adversarial_images_e=%d.npy' % (f, i))
				except:
					adv = np.load('FGSM_imgs/filter_double/%s/new_adversarial_images_e=%d.npy' % (f, i))
			imgs = filter_imgs(adv, f, params)
			accuracy = accuracy_score(model, imgs, labels)
			accuracies.append(accuracy)
			print(accuracy)
		all_accuracies.append(accuracies)
	np.save("all_accuracies/double_up/after/%s_%s.npy" % (attack, filter), all_accuracies)
	'''
	if not blackbox:
		np.save("for_thesis/PGD/cinic10_defense_testing_PGD.npy", all_accuracies)
	else:
		np.save("for_thesis/PGD/cinic10_defense_testing_PGD_BLACKBOX.npy", all_accuracies)
	'''

# shows filtered images
def filter_select(x_test, attack):
	filtering_methods = ["An Diff", "Med Gauss", "Nonlocal means", "TVD", "ROF"]
	advs = []
	advs.append(x_test[7432])
	eps = [5, 9]
	for e in eps:
		if attack == "FGSM":
			try:
				adv = np.load('FGSM_imgs/og_img_filter_network/subsample/adversarial_images_e=%d.npy' %  (e))
			except:
				adv = np.load('FGSM_imgs/og_img_filter_network/subsample/new_adversarial_images_e=%d.npy' %  (e))
			advs.append(adv[7432])
		elif attack == "PGD":
			adv = np.load('PGD_imgs/filters/%s/adversarial_images_e=%d.npy' % (f, e))
			advs.append(adv[7432])

	final_imgs = []
	final_imgs.append(advs)
	for f in filtering_methods:
		print(f)
		params = get_params(f)
		imgs = filter_imgs(advs, f, params)
		final_imgs.append(imgs)

	final_imgs = np.array(final_imgs)
	print(final_imgs.shape)
	final_imgs = final_imgs.reshape(18, 224, 224, 3)

	np.save('%s_imgs.npy', final_imgs)

	fig = plt.figure(1, figsize=(5, 8)) 
	plt.axis('off')
	grid = ImageGrid(fig, 111, nrows_ncols=(6, 3)) 

	for ax, im in zip(grid, final_imgs):
		ax.imshow(im/255)
		ax.axis('off')

	#plt.show()
	plt.savefig("for_thesis/%s/pipeline_compare_def_%s_1.png" % (attack, attack))


#subsample class examples with even distribution
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


def make_grid(x_test, y_test):
    ind = subsample(1000, y_test)
    imgs = x_test[ind]
    fig = plt.figure(1, figsize=(20, 10)) 
    plt.axis('off')
    grid = ImageGrid(fig, 111, nrows_ncols=(20, 50))
        
    for ax, im in zip(grid, imgs):
        ax.imshow(im/255)
        ax.axis('off')

    #plt.show()
    plt.savefig("img_grid.png")

def main():
	#model = load_model("/data1/share/cinic-10/vgg16_10_class.h5")
	#model = load_model("models/subsample_vgg16_10_HNN.h5")


	y_test = np.load('data/subsample_test_labels.npy')
	x_test = np.load('data/subsample_test_set.npy')
	make_grid(x_test, y_test)
	#filter_select(x_test, "FGSM")
	
	'''
	attacks = ["FGSM"]
	blackbox = False
	for a in attacks:
		if not blackbox:
			print("Running %s" % a)
		else:
			print("Running %s BLACKBOX" % a)
		model_list = []
		if a == "PGD":
			if not blackbox:
				model_list = ["Med Gauss", "Nonlocal means", "TVD", "ROF"]
			else:
				model_list = ["_BLACKBOX"]
		elif a == "FGSM":
			model_list = ["Med Gauss", "Nonlocal means", "TVD", "ROF"]
		for i in range(0, len(model_list)):
			model = None
			y_test = None
			x_test = None
			if not blackbox:
				model = load_model("models/%s_vgg16_10_HNN.h5" % model_list[i])
				y_test = np.load('data/%s_test_labels.npy' % model_list[i])
				x_test = np.load('data/%s_test_set.npy' % model_list[i])
			else:
				model = load_model("/data1/share/cinic-10/vgg16_10_class.h5")
				y_test = np.load('data/subsample_test_labels.npy')
				x_test = np.load('data/subsample_test_set.npy')
			print("Loaded in %s model" % model_list[i])		
			x_test = x_test.squeeze()
			x_test = x_test.astype('float32')
			for a in attacks:
				defend(model, x_test, y_test, a, model_list[i], blackbox)
	'''
if __name__ == "__main__":
	filtering_methods = ["Resize", "An Diff", "Med Gauss", "Nonlocal means", "TVD", "ROF"]
	main()

