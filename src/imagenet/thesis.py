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
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def img_test(index, adv_imgs, y_labels, x_imgs, model, attack, c, filter):

	#actual labels of the images
	img_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

	#initialize plot
	f = plt.figure(figsize=(10, 7))
	#label plot with truth class label
	#y = y_labels[index]
	#print(y)
	#plt.title("%s %s Example" % (attack, img_labels[y]))
	plt.axis('off')

	#add subplot
	f.add_subplot(1,2,1)
	plt.title("Adversarial Image")
	#choose an image to test
	og = x_imgs[index]
	#get prediction
	p1, c1 = prediction(model, og) 
	#reshape image for plot
	plt.imshow(og/255)
	#add annotation with prediction and confidence scores
	plt.annotate("Adversarial Image\nClass prediction: %s\nProbability: %.4f" % (img_labels[p1], c1), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

	'''
	#make second subplot
	f.add_subplot(1,3,2)
	#select image
	img = adv_imgs[index]
	difference = cv2.subtract(img, og)
	difference *= 10
	plt.title("Difference")
	plt.imshow(difference/255)
	'''

	img = adv_imgs[c]
	print(img.shape)
	#make third subplot
	f.add_subplot(1,2,2)
	p2, c2 = prediction(model, img)
	plt.title("%s Filtered Image" % filter)
	#reshape image for plot
	plt.imshow(img/255)
	#add annotation with prediction and confidence scores
	plt.annotate("Filtered Image\nClass prediction: %s\nProbability: %.4f" % (img_labels[p2], c2), (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	#plt.show()
	plt.savefig('for_thesis/%s/%s_PGD_example_img_%d.png' % (attack, filter, np.random.randint(100)), bbox_inches="tight")
	
#helper function to predict image class in img_test function
def prediction(model, image):
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


def accuracy_score(model, imgs, labels):
	pred = model.predict(imgs)
	pred = np.argmax(pred, axis=1)
	accuracy = metrics.accuracy_score(labels, pred)
	return accuracy


def attack(model, imgs, labels, attack, filter):

	accuracies = []

	print("Running %s" % attack)

	print("Step %d" % 0)
	accuracy = accuracy_score(model, imgs, labels)
	print(accuracy)
	accuracies.append(accuracy)

	for i in range(1, 10, 2):
		print("Step %d" % i)
		method = None
		adv = None
		if attack == "FGSM":
			method = fgsm(model, eps=i)
			adv = method.generate(imgs)
			np.save("FGSM_imgs/filter_double/%s/adversarial_images_e=%d.npy" % (filter, i), adv)
			print("Generated FGSM %s imgs" % filter)
			#adv = np.load('FGSM_imgs/og_img_filter_network/%s/adversarial_images_e=%d.npy' % (filter, i))
			#advs.append(adv)
		elif attack == "PGD":	
			#adv = np.load('PGD_imgs/new_adversarial_images_e=%d.npy' % i)
			#advs.append(adv)
			eps = i
			method = pgd(model, eps=eps, eps_step=eps*0.5, max_iter=2)
			adv = method.generate(imgs)
			np.save("PGD_imgs/filter_double/%s/adversarial_images_e=%d.npy" % (filter, i), adv)
			print("Generated PGD %s imgs" % filter)
		
		accuracy = accuracy_score(model, adv, labels)
		print(accuracy)
		accuracies.append(accuracy)
	np.save('for_thesis/%s/beaking_it_accuracy_%s.npy' % (attack, attack), accuracies)
			
	
def plot_results():
    arr = np.arange(0, 10) 
    #whitebox_fgsm = np.load('for_thesis/FGSM/base_accuracy_FGSM.npy')
    whitebox_pgd = np.load('for_thesis/PGD/base_accuracy_PGD.npy')
    #blackbox_pgd = np.load('for_thesis/PGD/base_accuracy_PGD_blackbox.npy')
    #blackbox_fgsm = np.load('for_thesis/FGSM/base_accuracy_FGSM_blackbox.npy')
    fig = plt.figure(figsize=(10,5))
    #plt.plot(arr, whitebox_fgsm, label="Whitebox FGSM")
    plt.plot(arr, whitebox_pgd, label="Whitebox PGD")
    #plt.plot(arr, blackbox_fgsm, label="Blackbox FGSM")
    #plt.plot(arr, blackbox_pgd, label="Blackbox PGD")
    plt.title("CINIC-10 Large White-Box PGD")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.show()
    plt.savefig('for_thesis/pgd_cinic10_attacks.png')

def compare_def(attack, box_type):
	numpy_name = ""
	if box_type == "Black Box":
		numpy_name = "_BLACKBOX"
	filters = ["Anisotropic Diffusion", "Median + Gaussian", "Nonlocal means", "TVD - Chambolle", "TVD - ROF"]
	arr = np.arange(0, 10)
	box = np.load('for_thesis/%s/base_accuracy_%s%s.npy' % (attack, attack, numpy_name.lower()))
	all_accuracies = np.load('for_thesis/%s/cinic10_defense_testing_%s%s.npy' % (attack, attack, numpy_name))[1:]
	fig = plt.figure()
	plt.plot(arr, box, '--', label="Base %s" % attack)
	for i in range(0, len(all_accuracies)):
		plt.plot(arr, all_accuracies[i], label=filters[i])
	plt.title("CINIC-10 Large %s %s Defenses Comparison" % (box_type, attack))
	plt.xlabel("Epsilon")
	plt.ylabel("Accuracy")
	plt.legend()
	#plt.show()
	plt.savefig('for_thesis/%s/cinic10_%s_def_comparison_%s.png' % (attack, attack, numpy_name))

def compare_trained(attack):

	filters = ["Anisotropic Diffusion", "Median + Gaussian", "Nonlocal means", "TVD - Chambolle", "TVD - ROF"]
	colors = ["b", "g", "r", "c", "m"]
	names = ["An Diff", "Med Gauss", "Nonlocal means", "TVD", "ROF"]


	x_axis = [0, 1, 3, 5, 7, 9]
	base = np.load('for_thesis/%s/base_accuracy_%s.npy' % (attack, attack))[x_axis]
	fig = plt.figure(figsize=(12, 7))
	all_data = np.load("for_thesis/%s/cinic10_defense_testing_%s.npy" % (attack, attack))[1:]
	
	'''
	for root, dirs, files in os.walk("all_accuracies/double_up/after"):
		for name in files:
			name_ = name[:-4].split("_")
			if name_[0] == attack:
				data = np.load("all_accuracies/double_up/after/%s" % name).reshape(-1, 1)
				plt.plot(x_axis, data, "--", label="%s Filtered Attack" % filters[names.index(name_[1])], color=colors[names.index(name_[1])])
	'''
	for i, data in enumerate(all_data):
		plt.plot(x_axis, data[x_axis], "--", label="Original %s Accuracy" % filters[i], color=colors[i])
	
	plt.plot(x_axis, base, "-.", label="Base %s" % attack, color="k")
	for root, dirs, files in os.walk("all_accuracies/after"):
		for name in files:
			name_ = name[:-4].split("_")
			if name_[0] == attack:
				data = np.load("all_accuracies/after/%s" % name).reshape(-1, 1)
				plt.plot(x_axis, data, label="With %s Training" % filters[names.index(name_[1])], color=colors[names.index(name_[1])])

	
	plt.legend(ncol=2)
	plt.title('%s Trained vs Original Defenses' % attack)
	plt.xlabel("Epsilon")
	plt.ylabel("Accuracy")
	plt.savefig("for_thesis/%s/filter_%s_trained_wb_comp.png" % (attack, attack))

def wrong_o(advs, model, y_test):
    ind_list = []
    for i in range(0, len(advs), 100):
        a = advs[i]
        l = y_test[i]
        pred = np.argmax(model.predict(a.reshape(1, 224, 224, 3)))
        truth = np.argmax(l)
        if truth != pred:
            ind_list.append(i)
            if len(ind_list) == 30: 
                return ind_list

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

# im so sorry for this i just wanted to be brain-dead while writing this section
def correct_o(advs, model, y_test):
	#ind_list_a = []
	#ind_list_b = []
	#ind_list_c = []
	ind_list_d = []
	ind_list_e = []
	#imgs_a = []
	#imgs_b = []
	#imgs_c = []
	imgs_d = []
	imgs_e = []
	for i in range(0, len(advs), 150):
		start = advs[i]
		truth = y_test[i]
		print(i)
		pred = np.argmax(model.predict(start.reshape(1, 224, 224, 3)))
		if truth == pred:
			continue
		print("Filtering")
		#a = filter_imgs_(start, "ROF", get_params("ROF"))
		#b = filter_imgs_(start, "Nonlocal means", get_params("Nonlocal means"))
		#c = filter_imgs_(start, "Med Gauss", get_params("Med Gauss"))
		d = filter_imgs_(start, "An Diff", get_params("An Diff"))
		e = filter_imgs_(start, "TVD", get_params("TVD"))
		print("Testing")
		#pred_a = np.argmax(model.predict(a.reshape(1, 224, 224, 3)))
		#pred_b = np.argmax(model.predict(b.reshape(1, 224, 224, 3)))
		#pred_c = np.argmax(model.predict(c.reshape(1, 224, 224, 3)))
		pred_d = np.argmax(model.predict(d.reshape(1, 224, 224, 3)))
		pred_e = np.argmax(model.predict(e.reshape(1, 224, 224, 3)))
		'''
		if truth == pred_a:
			print("Appending A")
			imgs_a.append(a)
			ind_list_a.append(i)
			print(len(ind_list_a))
		if truth == pred_b:
			print("Appending B")
			imgs_b.append(b)
			ind_list_b.append(i)
			print(len(ind_list_b))
		if truth == pred_c:
			print("Appending C")
			imgs_c.append(c)
			ind_list_c.append(i)
			print(len(ind_list_c))
		'''
		if truth == pred_d:
			print("Appending D")
			imgs_d.append(d)
			ind_list_d.append(i)
			print(len(ind_list_d))
		if truth == pred_e:
			print("Appending E")
			imgs_e.append(e)
			ind_list_e.append(i)
			print(len(ind_list_e))
		#if len(ind_list_a) > 10 and len(ind_list_b) > 10 and len(ind_list_c) > 10 and len(ind_list_d) > 10 and len(ind_list_e) > 10:
		#	print("Returning")
		#	return ind_list_a, ind_list_b, ind_list_c, ind_list_d, ind_list_e, imgs_a, imgs_b, imgs_c, imgs_d, imgs_e
		if len(ind_list_d) > 10 and len(ind_list_e) > 10:
			print("Returning")
			return ind_list_d, ind_list_e, imgs_d, imgs_e

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
	return filtered_img.reshape(224, 224, 3)


def main():
	#model = load_model("/data1/share/cinic-10/vgg16_10_class.h5")
	#print("ok")
	model = load_model("models/subsample_vgg16_10_HNN.h5")
	y_test = np.load('data/subsample_test_labels.npy')
	x_test = np.load('data/subsample_test_set.npy')
	x_test = x_test.squeeze()
	x_test = x_test.astype('float32')	
	#attack(model, x_test.squeeze(), y_test, a, None) 
	#plot_results()
	'''
	attacks = ["FGSM", "PGD"]
	for a in attacks:
		compare_trained(a)
	
	attacks = ['FGSM']
	for a in attacks:
		model_list = []
		if a == "PGD":
			model_list = ["An Diff", "Med Gauss", "Nonlocal means", "TVD", "ROF"]
		elif a == "FGSM":
			model_list = ["Nonlocal means", "TVD", "ROF"]
		for i in range(0, len(model_list)):
			model = load_model("models/%s_vgg16_10_HNN.h5" % model_list[i])
			model = KerasClassifier(model=model)
			print("Loaded in %s model" % model_list[i])
			y_test = np.load('data/%s_test_labels.npy' % model_list[i])
			x_test = np.load('data/%s_test_set.npy' % model_list[i])
			x_test = x_test.squeeze()
			x_test = x_test.astype('float32')
			for a in attacks:
				attack(model, x_test.squeeze(), y_test, a, model_list[i])
    
	model = KerasClassifier(model=model)
	eps = 5
	method = pgd(model, eps=eps, eps_step=eps*0.5, max_iter=5)
	advs = method.generate(x_test)
	np.save('PGD_imgs/new_adversarial_images_e=5.npy', advs)
	print("Done making advs")
	'''

	advs = np.load('PGD_imgs/new_adversarial_images_e=5.npy')
	#ind = wrong_o(advs, model, y_test)
	
	#ind_list_a, ind_list_b, ind_list_c, ind_list_d, ind_list_e, imgs_a, imgs_b, imgs_c, imgs_d, imgs_e = correct_o(advs, model, y_test)
	ind_list_d, ind_list_e, imgs_d, imgs_e = correct_o(advs, model, y_test)
	#np.save("indices.npy", np.array((ind_list_a, ind_list_b, ind_list_c, ind_list_d, ind_list_e)))
	#np.save("imgs.npy", np.array((imgs_a, imgs_b, imgs_c, imgs_d, imgs_e)))
	#ind_list_a, ind_list_b, ind_list_c, ind_list_d, ind_list_e = np.load("indices.npy", allow_pickle=True)
	#imgs_a, imgs_b, imgs_c, imgs_d, imgs_e = np.load("imgs.npy", allow_pickle=True)
	'''
	for i, c in enumerate(ind_list_a):
		img_test(c, imgs_a, y_test, advs, model, "PGD", i, "ROF")
	for i, c in enumerate(ind_list_b):
		img_test(c, imgs_b, y_test, advs, model, "PGD", i, "Nonlocal means")
	for i, c in enumerate(ind_list_c):
		img_test(c, imgs_c, y_test, advs, model, "PGD", i, "Med Gauss")
	'''
	for i, c in enumerate(ind_list_d):

		img_test(c, imgs_d, y_test, advs, model, "PGD", i, "An Diff")


	for i, c in enumerate(ind_list_e):

		img_test(c, imgs_e, y_test, advs, model, "PGD", i, "TVD")


	

if __name__ == "__main__":
	main()
