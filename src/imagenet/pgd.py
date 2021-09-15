from art.attacks import ProjectedGradientDescent
from art.classifiers import KerasClassifier
from keras.models import load_model
from keras import backend as k
from keras.utils import np_utils
import numpy as np
import os
from defense_testing import subsample
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

model = load_model("models/subsample_vgg16_10_HNN.h5")
print("Loaded in model")
model = KerasClassifier(model=model)

imgs = np.load("data/subsample_test_set.npy")
print("Loaded in %d images" % len(imgs))

t_labels = np.load("data/subsample_test_labels.npy")
print("Loaded in %d labels" % len(t_labels))

#extract 100 imgs
indices = subsample(1000, t_labels)
labels = np_utils.to_categorical(t_labels, 10) 

imgs = imgs[indices]
labels = labels[indices]

results = []

predictions = model.predict(imgs)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / len(labels)
results.append(accuracy)
print("Accuracy on original test examples: {}%".format(accuracy * 100))

for i in range(1, 11):
	pgd = ProjectedGradientDescent(classifier=model, eps=i)
	adv = pgd.generate(x=imgs)
	np.save("PGD_imgs/pgd_imgs_e=%d.npy" % i, adv)

	predictions = model.predict(adv)
	accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / len(labels)
	results.append(accuracy)
	print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

np.save("PGD_imgs/results.npy", results)

