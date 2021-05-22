#!/usr/bin/python3
# Doggy door Pre-trained demo from Nvidia DLI course
# https://modelzoo.co/
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)
    plt.show()

def load_and_process_image(image_path):
    print('Original image shape: ', mpimg.imread(image_path).shape)
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    print('Processed image shape: ', image.shape)
    return image

def readable_prediction(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    predictions = model.predict(image)
    print('Predicted:', decode_predictions(predictions, top=3))

def doggy_door(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    preds = model.predict(image)
    if 151 <= np.argmax(preds) <= 268: print("Doggy come on in!")
    elif 281 <= np.argmax(preds) <= 285: print("Kitty stay inside!")
    else: print("You're not a dog! Stay outside!")

if __name__ == '__main__':
    model = VGG16(weights="imagenet")
    model.summary()
    processed_image = load_and_process_image("doggy_door_images/brown_bear.jpg")

    readable_prediction("doggy_door_images/happy_dog.jpg")
    readable_prediction("doggy_door_images/brown_bear.jpg")
    readable_prediction("doggy_door_images/sleepy_cat.jpg")

    doggy_door("doggy_door_images/brown_bear.jpg")
    doggy_door("doggy_door_images/happy_dog.jpg")
    doggy_door("doggy_door_images/sleepy_cat.jpg")