import matplotlib.pyplot as plt
import numpy as np
import sys

def plot(both, name):
	
	acc = np.load("overall/accuracies.npy")
	x_axis = np.arange(0, len(acc))
	untrained_acc = np.load("untrained_%s.npy" % name)

	if both:
		trained_acc = np.load("trained_%s.npy" % name)

	plt.plot(x_axis, acc, 'b', label="Adversarial Accuracy")
	plt.plot(x_axis, untrained_acc, 'r', label="Untrained clean Accuracy")
	if both:
		plt.plot(x_axis, trained_acc, 'g', label="Trained clean Accuracy")
	plt.legend()
	plt.xlabel("Epsilon")
	plt.ylabel("Accuracy")
	plt.title("Total Variation Denoising defense comparison FGSM")
	save_name = "%s_defense" % name
	if both:
		save_name += "1"
	plt.savefig("results/TVD/%s_FGSM.png" % save_name)

def plot_overall():
	
	fig = plt.figure(figsize=(20,10))

	adv_acc = [0.88, 0.62, 0.39, 0.25, 0.18, 0.144, 0.12, 0.11, 0.11, 0.10, 0.10]
	before_rof = np.load("whitebox/clean/before_overall_ROF.npy", allow_pickle=True)
	before_nlm = np.load("whitebox/clean/before_overall_Nonlocal means.npy", allow_pickle=True)
	before_mg = np.load("whitebox/clean/before_overall_Med Gauss.npy", allow_pickle=True)
	before_ad = np.load("whitebox/clean/before_overall_An Diff.npy",  allow_pickle=True)
	before_tvd = np.load("whitebox/clean/before_overall_TVD.npy",  allow_pickle=True)

	# put one up here that shows ideal
	after_rof = np.load("whitebox/clean/after_overall_ROF.npy", allow_pickle=True)
	after_nlm = np.load("whitebox/clean/after_overall_Nonlocal means.npy", allow_pickle=True)
	after_mg = np.load("whitebox/clean/after_overall_Med Gauss.npy", allow_pickle=True)
	after_ad = np.load("whitebox/clean/after_overall_An Diff.npy",  allow_pickle=True)
	after_tvd = np.load("whitebox/clean/after_overall_TVD.npy",  allow_pickle=True)

	x_axis = np.arange(0, len(after_tvd))
	
	plt.plot(x_axis, before_rof, 'b--', label="ROF adv Accuracy")
	plt.plot(x_axis, before_nlm, 'g--', label="Nonlocal Means adv Accuracy")
	plt.plot(x_axis, before_mg, 'r--', label="Median + Gaussian adv Accuracy")
	plt.plot(x_axis, before_ad, 'c--', label="Anisotropic Diffusion adv Accuracy")
	plt.plot(x_axis, before_tvd, 'm--', label="Total Variation adv Accuracy")
	
	orig = plt.plot(x_axis, adv_acc, 'k-.', label="Clean Adversarial Accuracy")
	
	plt.plot(x_axis, after_rof, 'b', label="ROF adv filtered Accuracy")
	plt.plot(x_axis, after_nlm, 'g', label="Nonlocal Means adv filtered Accuracy")
	plt.plot(x_axis, after_mg, 'r', label="Median + Gaussian adv filtered Accuracy")
	plt.plot(x_axis, after_ad, 'c', label="Anisotropic Diffusion adv filtered Accuracy")
	plt.plot(x_axis, after_tvd, 'm', label="Total Variation adv filtered Accuracy")

	plt.legend(fontsize="small", ncol=2)
	plt.xlabel("Epsilon")
	plt.ylabel("Accuracy")
	plt.title("White Box (Option 1) defenses comparison FGSM")
	plt.savefig("results/overall_option1_whitebox_FGSM.png")

def examine_grid():

	#130
	og_a = np.load("og_a_1.npy")
	adv_a = np.load("adv_a_1.npy")
	params = np.load("params_1.npy")

	print(og_a)
	print(adv_a)
	print(params)

	max = 0
	index = None
	for i in range(0, len(adv_a)):
		acc = adv_a[i][-1]
		if acc > max:
			max = acc
			index = i
	print(params[index])

def plot_per_class():

	names = ["An Diff", "Med Gauss", "Nonlocal means", "ROF", "TVD"]
	class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	#before_overall = []
	after_overall = []
	for n in names:
		#classes = np.load("whitebox/clean_per_class/before_per_class_%s.npy" % n, allow_pickle=True)
		#before_overall.append(classes)
		classes = np.load("whitebox/clean_per_class/after_per_class_%s.npy" % n, allow_pickle=True)
		after_overall.append(classes)
	
	for e in range(0, 10):
		#b_bar_values = []
		a_bar_values = []
		for i in range(0, len(after_overall)):
			method_name = names[i]
			# which epsilon we on
		#	b_bar_values.append(before_overall[i][e])
			a_bar_values.append(after_overall[i][e])
		# plot it
		#b_bar_values = np.array(b_bar_values)
		a_bar_values = np.array(a_bar_values)
		fig, ax = plt.subplots(figsize=(20,10))
		ind = np.arange(len(class_labels))
		width = 0.1
		
		#bad = ax.bar(ind, b_bar_values[0], width)
		aad = ax.bar(ind, a_bar_values[0], width) #, bottom=b_bar_values[0])
		#bmg = ax.bar(ind + width, b_bar_values[1], width)
		amg = ax.bar(ind + width, a_bar_values[1], width) #, bottom=b_bar_values[1])
		#bnlm = ax.bar(ind + width * 2, b_bar_values[2], width)
		anlm = ax.bar(ind + width * 2, a_bar_values[2], width) # , bottom=b_bar_values[2])
		#btvd = ax.bar(ind + width * 3, b_bar_values[3], width)
		atvd = ax.bar(ind + width * 3, a_bar_values[3], width) #, bottom=b_bar_values[3])
		#brof = ax.bar(ind + width * 4, b_bar_values[4], width)
		arof = ax.bar(ind + width * 4, a_bar_values[4], width) #, bottom=b_bar_values[4])
		
		ax.set_ylabel("Correctly classified examples out of 1,000", fontsize=14)
		ax.set_xticks(ind + width * 2)
		ax.tick_params(labelsize=14)
		ax.set_xticklabels(class_labels)
		ax.legend((aad[0], amg[0], anlm[0], arof[0], atvd[0]), 
				  ("Anisotropic Diffusion", "Median + Gaussian", "Nonlocal Means", "ROF", "Total Variation Denoising"), ncol=2)
		ax.set_title("Classes correctly classified with epsilon = %d" % e, fontsize=14)
		plt.savefig("whitebox/clean_per_class/class_graph_%d.png" % e)

if __name__ == "__main__":

	plot_per_class()
	#both = True
	#name = "TVD"
	#plot(both, name)
	#plot_overall()
	#examine_grid()
