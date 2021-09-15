#program to make image arrays and image directories

import shutil
import numpy as np
import os
import re
import cv2
import random
import sys
sys.path.append("..")
from filters import filters
import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt


def make_arrays(dest, train=False):
	
	if train:
		make_arrays_helper(dest, "train")
	make_arrays_helper(dest, "valid")


def make_arrays_helper(dest, set_name):
	classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	imgs = []
	labels = []

	path = ("%s/%s" % (dest, set_name))
	for root, dirs, files in os.walk(path):
		for name in files:
			class_name = re.search(r"(.*)_", name).group(1)
			label = classes.index(class_name)
			print(label)
			img_path = os.path.join(root, name)
			img = Image.open(img_path)
			#skip over if there was an error reading the image
			if img is None:
				continue
			img = np.asarray(img, dtype=np.float32)
			img = img/255
			imgs.append(img)
			labels.append(label)

	if set_name == "valid":
		set_name = "test"
	np.save("../data/%s_%s_set_BUG.npy" % (dest, set_name), imgs)
	np.save("../data/%s_%s_labels_BUG.npy" % (dest, set_name), labels)

	print("Finished %s set" % set_name)



def make_dirs(src, dest):

	train_set = np.load("../data/%s_train_set.npy" % src)
	test_set = np.load("../data/%s_test_set.npy" % src)
	train_labels = np.load("../data/%s_train_labels.npy" % src)
	test_labels = np.load("../data/%s_test_labels.npy" % src)
	print("Loaded in data")

	if os.path.exists(dest):
		shutil.rmtree(dest)
	
	os.mkdir(dest)

	if not os.path.exists("%s/train" % dest):
		os.mkdir("%s/train" % dest)

	# only use this when making a brand new data set
	train_indices = None
	if dest == "subsample":
		train_indices = subsample(30000, train_labels)
	
	print("Filtering the training data")
	train_set = choose_filter(train_set, dest)
	print("Done")

	make_imgs(train_set, dest, train_labels, train_indices, train=True)
	
	if not os.path.exists("%s/valid" % dest):
		os.mkdir("%s/valid" % dest)
	
	test_indices = None
	if dest == "subsample":
		test_indices = subsample(10000, test_labels)
	
	print("Filtering the testing data")
	test_set = choose_filter(test_set, dest)
	print("Done")
	
	make_imgs(test_set, dest, test_labels, test_indices)

def make_imgs(set, dest, labels, indices, train=False):
		classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
		iterator = None
		if indices:	
			iterator = indices
		else:
			iterator = set
		for i in range(0, len(iterator)):
			if i % 1000 == 0:
				print("%d out of %d imgs done" % (i, len(iterator)))
			if indices:
				curr_class = classes[labels[iterator[i]]]
				img = set[iterator[i]]
			else:
				curr_class = classes[labels[i]]
				img = set[i]
			if len(img.shape) >= 4:
				img = np.squeeze(img, axis=0)
			if indices:
				img = img[:,:,::-1]
			#rescale image to be between 0-255 and uint8
			img = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
			img = Image.fromarray(img)
			if indices:
				img = img.resize((224, 224), Image.ANTIALIAS)
			#doing this to ensure every image has a unique name
			num = random.randint(0, 100000000)
			name = "%s_%d" % (curr_class, num)
			dir = "train" if train else "valid"
			if not os.path.exists("%s/%s/%s" % (dest, dir, curr_class)):
				os.mkdir("%s/%s/%s" % (dest, dir, curr_class))
			img.save("%s/%s/%s/%s.png" % (dest, dir, curr_class, name))
		print("Wrote %s images to file" % dir)


def choose_filter(set, method):
	if method == "ROF":
		set = filter_imgs(set, method, 65)
	elif method == "Nonlocal means":
		set = filter_imgs(set, method, 10, 60)
	elif method == "Med Gauss":
		set = filter_imgs(set, method, 9, 1.5, 5)
	elif method == "An Diff":
		set = filter_imgs(set, method, 10, 400, 0.25, 2)
	elif method == "TVD":
		set = filter_imgs(set, method, 130)
	return set
	

def filter_imgs(imgs, method, one, two=None, three=None, four=None):

	pool = mp.Pool(int(mp.cpu_count()/5))
	filtered_imgs = pool.starmap(filter_imgs_, [(imgs[i], method, one, two, three, four) for i in range(0, len(imgs))])
	pool.close()
	return filtered_imgs


def filter_imgs_(img, method, one, two, three, four):

	i = img[:,:,::-1]
	filtered_img = None
	if method == "ROF":
		filtered_img = filters.ROF(i, one)
	elif method == "Nonlocal means":
		filtered_img = filters.nonlocal_means(i, one, two, 10, 5, fast=True)
	elif method == "Med Gauss":
		filtered_img = filters.median(i, one)
		filtered_img = filters.gaussian(filtered_img, two, three)
	elif method == "An Diff":
		filtered_img = filters.anisotropic_diffusion(i, one, two, three, four)
	elif method == "TVD":
		filtered_img = filters.TVD(i, one)
	return filtered_img


#function to subsample class examples with even distribution
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


if __name__ == "__main__":
	
	
	make_arrays("An Diff")
	
	'''
	src = "subsample" 
	
	print("Making %s array" % src)
	make_arrays(src)
	dests = ["An Diff", "Med Gauss", "Nonlocal means", "TVD", "ROF"]
	
	# make_dirs("raw", "subsample", method="subsample")

	for dest in dests:
		# print("Making %s dir" % dest)
		# make_dirs(src, dest)
		print("Making %s array" % dest)
		make_arrays(dest)
	'''
