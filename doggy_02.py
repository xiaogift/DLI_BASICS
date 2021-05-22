#!/usr/bin/python3
# Doggy door Transfer learning demo from Nvidia DLI course
# https://modelzoo.co/
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)
base_model.summary()
base_model.trainable = False

# Separately from setting trainable on the model, we set training to ase 
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)

# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()

# Important to use binary crossentropy and binary accuracy as we now have a binary classification problem
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

datagen = ImageDataGenerator( featurewise_center=True,  # set input mean to 0 over the dataset
                              samplewise_center=True,  # set each sample mean to 0
                              rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                              zoom_range = 0.1, # Randomly zoom image 
                              width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                              height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                              horizontal_flip=True,  # randomly flip images
                              vertical_flip=False) # we don't expect Bo to be upside-down so we will not flip vertically

# load and iterate training dataset
train_it = datagen.flow_from_directory('presidential_doggy_door/train/', 
                                       target_size=(224, 224), 
                                       color_mode='rgb', 
                                       class_mode='binary', 
                                       batch_size=8)
# load and iterate test dataset
test_it = datagen.flow_from_directory('presidential_doggy_door/test/', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode='binary', 
                                      batch_size=8)

model.fit(train_it, steps_per_epoch=12, validation_data=test_it, validation_steps=4, epochs=20)

# Unfreeze the base model
base_model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes are taken into account
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(train_it, steps_per_epoch=12, validation_data=test_it, validation_steps=4, epochs=10)

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)

def make_predictions(image_path):
    show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds

make_predictions('presidential_doggy_door/test/bo/bo_20.jpg')
make_predictions('presidential_doggy_door/test/not_bo/121.jpg')

def presidential_doggy_door(image_path):
    preds = make_predictions(image_path)
    if preds[0] < 0: print("It's Bo! Let him in!")
    else: print("That's not Bo! Stay out!")

presidential_doggy_door('presidential_doggy_door/test/not_bo/131.jpg')
presidential_doggy_door('presidential_doggy_door/test/bo/bo_29.jpg')

# loss: 8.3007e-06 - binary_accuracy: 1.0000 - val_loss: 0.0060 - val_binary_accuracy: 1.0000
# loss: 0.0034 - binary_accuracy: 1.0000 - val_loss: 0.1294 - val_binary_accuracy: 0.9667