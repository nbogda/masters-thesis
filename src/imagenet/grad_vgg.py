from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, LeakyReLU
from keras.regularizers import l2

model = Sequential()
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', ))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Add Fully Connected Layers
model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(512, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.13)))
model.add(LeakyReLU(alpha=0.05))


model.add(Dropout(0.5))
model.add(Dense(512, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.13)))
model.add(LeakyReLU(alpha=0.05))

model.add(Dense(10, activation='softmax'))

n_model = load_model("/data1/share/cinic-10/vgg16_10_class.h5")

for i in range(len(model.layers[:-7])):
    model.layers[i].set_weights(n_model.layers[i].get_weights())
    
del n_model

for layer in model.layers[:-7]:
    layer.trainable = False

op = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss = "categorical_crossentropy", optimizer = op, metrics=["accuracy"])