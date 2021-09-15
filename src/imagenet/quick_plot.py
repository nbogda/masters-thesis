import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model

model = load_model("models/HNN_FOR_PAPER.h5")
plot_model(model, show_shapes=True, to_file='graphs/vgg16_10_HNN.png')
