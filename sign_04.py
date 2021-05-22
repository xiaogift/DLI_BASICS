#!/usr/bin/python3
# American sign Inference demo from Nvidia DLI course
from tensorflow import keras
from tensorflow.keras.preprocessing import image as image_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.show()

def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode="grayscale", target_size=(28,28))
    return image

def predict_letter(file_path):
    show_image(file_path)
    image = load_and_scale_image(file_path)
    image = image_utils.img_to_array(image)
    image = image.reshape(1,28,28,1) 
    image = image/255
    prediction = model.predict(image)
    alphabet = "abcdefghiklmnopqrstuvwxy"
    dictionary = {}
    for i in range(24): dictionary[i] = alphabet[i]
    predicted_letter = dictionary[np.argmax(prediction)]
    return predicted_letter

if __name__ == "__main__":
    model = keras.models.load_model('asl_model')
    model.summary()
    predict_letter("asl_test/b.png")
    predict_letter("asl_test/a.png")