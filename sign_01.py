#!/usr/bin/python3
# American sign initial demo from Nvidia DLI course
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os, pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# https://www.kaggle.com/datamunge/sign-language-mnist
train_df = pd.read_csv("asl_data/sign_mnist_train.csv")
test_df = pd.read_csv("asl_data/sign_mnist_test.csv")
train_df.head()

y_train = train_df['label']
y_test = test_df['label']

del train_df['label']
del test_df['label']

x_train = train_df.values
x_test = test_df.values
x_train = x_train / 255
x_test = x_test / 255

num_classes = 26 # 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(units = 512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_test, y_test))

# loss: 0.1098 - accuracy: 0.9779 - val_loss: 2.3473 - val_accuracy: 0.7823
# loss: 1.9047 - accuracy: 0.3494 - val_loss: 5.5414 - val_accuracy: 0.0602 # 24