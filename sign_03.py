#!/usr/bin/python3
# American sign Augmentation demo from Nvidia DLI course
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_df = pd.read_csv("asl_data/sign_mnist_train.csv")
test_df = pd.read_csv("asl_data/sign_mnist_test.csv")

y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

x_train = train_df.values
x_test = test_df.values
x_train = x_train / 255
x_test = x_test / 255

num_classes = 26
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))
model.summary()

# randomly rotate images in the range (degrees, 0 to 180)
# Randomly zoom image 
# randomly shift images horizontally (fraction of total width)
# randomly shift images vertically (fraction of total height)
# randomly flip images horizontally
# Don't randomly flip images vertically
datagen = ImageDataGenerator( rotation_range=10,
                              zoom_range = 0.1,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              horizontal_flip=True,
                              vertical_flip=False)
datagen.fit(x_train)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Default batch_size is 32.
# Run same number of steps we would if we were not using a generator.
model.fit( datagen.flow(x_train,y_train, batch_size=32), 
           epochs=20,
           steps_per_epoch=len(x_train)/32,
           validation_data=(x_test, y_test))
model.save('asl_model')

# loss: 0.0500 - accuracy: 0.9863 - val_loss: 0.2020 - val_accuracy: 0.9561
