import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model

model = load_model("model/OG_cifar10_resnet.h5")
plot_model(model, show_shapes=True, to_file='graphs/cifar10_resnet_struct.png')
