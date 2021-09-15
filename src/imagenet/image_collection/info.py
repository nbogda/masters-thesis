import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random

def count_files(dir):

	sets = ["train", "valid"]
	train_dict = {}
	val_dict = {}
	dicts = [train_dict, val_dict]

	os.chdir(dir)
	for i in range(0,2):
		for root, dirs, files in os.walk(sets[i]):
			curr_class = os.path.basename(root)
			if curr_class not in sets:
				dicts[i][curr_class] = len(files)
		
	vals = [] 
	for i in range(0, 2):
		vals.append(np.array(list(dicts[i].values())))
	
	plot_dist(vals, dicts)


'''
#using function to jumble labels
def move_files():

	class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship    ", "truck"]
	sets = ["train", "valid"]
	for i in range(0, len(sets)):
		for root, dirs, files in os.walk(sets[i]):
			curr_class = os.path.basename(root)
			if curr_class not in sets:
				for file in files:
					#get a random class from array
					random_index = random.randint(0, 9)
					dst_path = "mix_%s/%s" % (sets[i], class_names[random_index])
					if not os.path.exists(dst_path):
						os.mkdir(dst_path)
					src_path = os.path.join(root, file)
					shutil.copy(src_path, dst_path)
'''
	
def plot_dist(vals, dicts):

	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	ind = np.arange(len(dicts[0].keys()))

	
	train = plt.bar(ind, vals[0], color='r')
	valid = plt.bar(ind, vals[1], bottom = vals[0], color='b')

	plt.xticks(ind, dicts[0].keys())
	plt.yticks()
	plt.title("Class Distribution")
	plt.legend((train, valid), ("Train set", "Validation set"))
	#plt.show()
	plt.savefig("Class_Dist.png")


if __name__ == "__main__":

	dir = sys.argv[1]
	count_files(dir)

