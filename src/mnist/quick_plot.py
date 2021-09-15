import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from collections import OrderedDict
from keras.utils import plot_model
from keras.models import load_model

def plot():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	count_dict = {}
	for y in y_test:
		if y not in count_dict:
			count_dict[y] = 1
		else:
			count_dict[y] += 1

	count_dict = OrderedDict(sorted(count_dict.items()))

	fig, ax = plt.subplots()
	x = np.arange(len(count_dict.keys()))
	plt.bar(x, count_dict.values())
	plt.title("Distribution of MNIST testing set")
	plt.ylabel("Count")
	plt.xlabel("Classes")
	ax.set_xticks(x)
	ax.set_xticklabels(count_dict.keys())
	plt.savefig("graphs/testing_set_dist.png")

def plot_m():
	model = load_model("model/mnist_model.h5")
	plot_model(model, show_shapes=True, to_file='graphs/mnist_model.png')

def plot_compare():
	arr = [0, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
	#arr = np.array(arr)/255
	#whitebox_fgsm = np.load('results/FGSM/base_accuracy.npy')
	whitebox_pgd = np.load('results/PGD/base_accuracy.npy')
	print(whitebox_pgd)
	#blackbox_pgd = np.load('results/PGD/blackbox.npy')
	#blackbox_fgsm = np.load('results/FGSM/blackbox.npy')
	fig = plt.figure(figsize=(10, 5))
	#plt.plot(arr, whitebox_fgsm, label="Whitebox FGSM")
	#plt.plot(arr, blackbox_fgsm, label="Blackbox FGSM")
	plt.plot(arr, whitebox_pgd, label="Whitebox PGD")
	#plt.plot(arr, blackbox_pgd, label="Blackbox PGD")
	plt.title("MNIST White-Box PGD")
	plt.xlabel("Epsilon")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.savefig('mnist_pgd_only.png')


def plot_compare_def():
	attack = "PGD"
	filters = ["Resized", "Thresholded", "Gaussian", "Median", "Bilateral"]
	arr = [0, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
	#arr = np.array(arr)/255
	whitebox = np.load('results/%s/base_accuracy.npy' % attack)
	all_accuracies = np.load('results/%s/defense_testing.npy' % attack)
	#blackbox = np.load('results/%s/blackbox.npy' % attack)
	fig = plt.figure()
	plt.plot(arr, whitebox, '--', label="Base %s" % attack)
	for i in range(0, len(all_accuracies)):
		plt.plot(arr, all_accuracies[i], label=filters[i])
		print(all_accuracies[i], filters[i])
	plt.title("MNIST Whitebox %s Defenses Comparison" % attack)
	plt.xlabel("Epsilon")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.savefig('%s_mnist_def_comparison.png' % attack)



def main():
	#plot()
	#plot_m()
	#plot_compare()
	plot_compare_def()

if __name__ == "__main__":
	main()
