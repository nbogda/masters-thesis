from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2 
import PIL.Image
import urllib
import os
import matplotlib.pyplot as plt
import pandas as pd
import csv
import re
import codecs

def get_image(row, url):

	#img_rows = 256
	#img_cols = 256
	
	synset = row.iloc[0,0]
	image_num = str(row.iloc[0,1]).strip()
	set = row.iloc[0,2].strip()
	class_label = row.iloc[0,3].strip()

	print("Getting %s image." % class_label)

	path = "originals/%s/%s" % (set, class_label)

	if not(os.path.exists(path)):
		os.mkdir(path)
	resp = None
	try:
		resp = urllib.request.urlopen(url, timeout=50)
	except (urllib.error.URLError, urllib.error.HTTPError, urllib.error.ContentTooShortError):
		pass
	except Exception as e:
		pass
	if resp:
		try:
			image = np.asarray(bytearray(resp.read()), dtype="uint8")
			image = cv2.imdecode(image, cv2.IMREAD_COLOR)
			#image = cv2.resize(image, (img_rows, img_cols), interpolation = cv2.INTER_AREA)
			save_path = "%s/%s_%s_%s.jpg" % (path, class_label, synset, image_num)
			cv2.imwrite(save_path, image)
		except:
			pass


def df_urls(file_name, df):

	min_s = min(df['synset'])
	max_s = max(df['synset'])
	
	txt = open(file_name, 'r', encoding='iso-8859-1')
	f = txt.readlines()
	fall_dict = {}
	for i, line in enumerate(f):
		if i % 1000 == 0:
			print("%d out of %d lines checked" % (i, len(f)))
		line = line.strip('\n').split('\t')
		wnid = int(re.search(r'^n(\d\d\d\d\d\d\d\d)_(.*)$', line[0]).group(1))
		if wnid <= max_s and wnid >= min_s:
			if wnid in df.synset.values:
				image_num = int(re.search(r'^(n\d\d\d\d\d\d\d\d)_(.*)$', line[0]).group(2))
				sub = df.loc[(df.synset == wnid)]
				if image_num >= min(sub.image_num.values) and image_num <= max(sub.image_num.values):
					if image_num in sub.image_num.values:
						row = df.index[(df.synset == wnid) & (df.image_num == image_num)]
						get_image(df.iloc[row,], line[1])
	
#this is because i wanted to look at the data i was going to be working with
def plot_dist(column, set):

	train_dict = {}
	val_dict = {}
	test_dict = {}
	for c, s in zip(column, set):
		if s == " train":
			if c not in train_dict:
				train_dict[c] = 1
			else:
				train_dict[c] += 1
		elif s == " valid":
			if c not in val_dict:
				val_dict[c] = 1
			else:
				val_dict[c] += 1
		else:
			if c not in test_dict:
				test_dict[c] = 1
			else:
				test_dict[c] += 1


	train_d = np.array(list(train_dict.values()))
	val_d = np.array(list(val_dict.values()))
	test_d = np.array(list(test_dict.values()))

	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	ind = np.arange(len(train_dict.keys()))
	
	train = plt.bar(ind, train_d, color='r')
	val = plt.bar(ind, val_d, bottom = train_d, color='g')
	test = plt.bar(ind, test_d, bottom = train_d + val_d, color='b')
	
	plt.xticks(ind, train_dict.keys())
	plt.yticks()
	plt.title("Class distribution")
	plt.legend((train, val, test), ("Train set", "Validation set", "Test set"))
	plt.savefig("Initial_class_dist.png")


if __name__ == "__main__":

	sets = ['train', 'valid', 'test']
	classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


	#column names: synset, image_num, cinic_set, class
	df = pd.read_csv("txtfiles/imagenet-contributors.csv")
	df.columns=["synset", "image_num", "set", "class"]
	synset = df['synset'].tolist()
	for i in range(0, len(synset)):
		synset[i] = int(synset[i][1:])
	df['synset'] = synset
	
	df_urls("txtfiles/fall11_urls.txt", df)

	#plot_dist(df["class"], df["cinic_set"])

