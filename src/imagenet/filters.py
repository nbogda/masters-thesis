#this class contains a bunch of filters i am going to try to use to filter out adversarial noise
#the purpose of this class is to bring all filters to one central location, imports and all
from keras.utils import np_utils
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
from medpy.filter import smoothing
import warnings
from scipy.ndimage import filters as scipy_filters
from keras.models import load_model
from keras import backend as k
warnings.filterwarnings("ignore")
#to make tensorflow shut up
import os
import sys
import cv2
import tensorflow.compat.v1 as tf
from skimage import data, color, img_as_float
from skimage.restoration import denoise_tv_chambolle
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

class filters:

	#apply non local means filter
	def nonlocal_means(img, sigma_est, h_val, size, distance, fast=False):
		patch_kw = dict(patch_size=size, patch_distance=distance, multichannel=True)
		denoise = denoise_nl_means(img, h=h_val * sigma_est, sigma=sigma_est, fast_mode=fast, **patch_kw)
		return denoise

	#apply anisotropic diffusion filter
	def anisotropic_diffusion(img, K, niters):
		denoise = cv2.ximgproc.anisotropicDiffusion(img.astype('uint8'), alpha=0.25, K=K, niters=niters)
		return denoise

	#apply median filter
	def median(img, size):
		denoise = cv2.medianBlur(img, size)
		return denoise

	#apply gaussian filter
	def gaussian(img, sigma, size):
		denoise = cv2.GaussianBlur(img, (size, size), sigma)
		return denoise
	
	#rudin, osher, and fatemi algorithm
	def ROF(img, weight, eps=1e-3, num_iter_max=200):
		#note: i did not write this code, this is copy and pasted off of github
		#https://gist.github.com/mbeyeler/d9c4cff18e8b7324cd0f319d2841e72c
		u = np.zeros_like(img)
		px = np.zeros_like(img)
		py = np.zeros_like(img)
		nm = np.prod(img.shape[:2])
		tau = 0.125
		i = 0
		while i < num_iter_max:
			u_old = u
			# x and y components of u's gradient
			ux = np.roll(u, -1, axis=1) - u
			uy = np.roll(u, -1, axis=0) - u
			# update the dual variable
			px_new = px + (tau / weight) * ux
			py_new = py + (tau / weight) * uy
			norm_new = np.maximum(1, np.sqrt(px_new **2 + py_new ** 2))
			px = px_new / norm_new
			py = py_new / norm_new
			# calculate divergence
			rx = np.roll(px, 1, axis=1)
			ry = np.roll(py, 1, axis=0)
			div_p = (px - rx) + (py - ry)
			# update image
			u = img + weight * div_p
			# calculate error
			error = np.linalg.norm(u - u_old) / np.sqrt(nm)
			if i == 0:
				err_init = error
				err_prev = error
			else:
				# break if error small enough
				if np.abs(err_prev - error) < eps * err_init:
					break
				else:
					e_prev = error
			# don't forget to update iterator
			i += 1
		return u	


	def TVD(img, weight):
		img = img.astype(np.float32)
		denoise = denoise_tv_chambolle(img, weight=weight, multichannel=True)
		return denoise

	#show the filtered image
	def show_img(img, title=None):
		if len(img.shape) > 3:
			img = np.squeeze(img)
		img = np.clip(img, 0, 255)
		img = img / 255
		plt.imshow(img[...,::-1])
		if title:
			plt.title(title)
		plt.show()


#helper function to predict image class in img_test function
def prediction(model, img):
	'''
	model : keras Sequential model 
			neural network object to prediction with
	img : image of 224 by 224 by 3
		the image to predict
	'''
	#reshape image for keras
	if (len(img.shape) < 4):
		img = np.expand_dims(img, axis=0)
	#make prediction
	pred = model.predict(img)
	#get index of class label
	p = np.argmax(pred[0])
	#get confidence score
	c = max(pred[0])
	#return prediction label and confidence score
	return p, c

def test_imgs(model, imgs, labels):

	accuracy = 0 
	avg_confidence = 0 

	for i in range(0, len(labels)):
		x = imgs[i]
		#check to see if image is 4 dimensional
		if len(x.shape) != 4:
			#if not, wrap it in one dimension
			x = np.expand_dims(x, axis=0)
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

if __name__ == "__main__":

	all_imgs = np.load("data/subsample_test_set.npy")
	img = all_imgs[1000]
	if len(img.shape) >= 4:
		img = np.squeeze(img, axis=0)
	img = img[:,:,::-1]
	img = filters.gaussian(img, 1.5, 5)
	img = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
	img = Image.fromarray(img)
	# img = img.resize((224, 224), Image.ANTIALIAS)
	img.save("TEST_IMAGE.png")
	

	
