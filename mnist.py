#!/usr/bin/python3
# MNIST demo from Nvidia DLI course
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
import os, random, matplotlib.pyplot as plt

# https://www.tensorflow.org/install/gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image = x_train[random.randint(0,len(x_train))]
plt.imshow(image, cmap='gray'); plt.show()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train / 255
x_test = x_test / 255 

num_categories = 10
y_train = keras.utils.to_categorical(y_train, num_categories)
y_test = keras.utils.to_categorical(y_test, num_categories)

model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units= 10, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test,y_test))