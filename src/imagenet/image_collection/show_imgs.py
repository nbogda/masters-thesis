import numpy as np
import matplotlib.pyplot as plt
from make_img_arrays import subsample
from mpl_toolkits.axes_grid1 import ImageGrid

imgs = np.load("../data/raw_test_set.npy")
labels = np.load("../data/raw_test_labels.npy")

indices = subsample(10, labels)
img_subset = imgs[indices]
label_subset = labels[indices]
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

fig = plt.figure(1, figsize=(4, 8))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 2), axes_pad = 0.4)

for ax, i in zip(grid, np.arange(0, len(img_subset))):
	img = np.squeeze(img_subset[i])
	ax.imshow(img[...,::-1])
	ax.set_title(classes[label_subset[i]], loc='center')

plt.savefig("cinic_10_large_ex.png")

