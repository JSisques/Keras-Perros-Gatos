import matplotlib as plt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as kr
import cv2
from PIL import Image

from keras.preprocessing.image import load_img, img_to_array
from numpy.core.defchararray import array
from os import listdir


path_test_cats = './Test/Cat/'
path_test_dogs = './Test/Dog/'

model = kr.models.load_model('model.h5')

photos_cats = []
photos_dogs = []

for file in listdir(path_test_cats):
    print('Gatos: ', file)

    photo = load_img(path_test_cats +  file, target_size=(150, 150))
    photo = img_to_array(photo)

    
    photos_cats.append(photo)


photos_cats = np.array(photos_cats)

image = Image.fromarray((photos_cats[7]).astype(np.uint8))
image.show()


photos_cats = photos_cats / 255



print(photos_cats[7])

predict = model.predict(np.expand_dims(photos_cats[7], 0))
print(predict)