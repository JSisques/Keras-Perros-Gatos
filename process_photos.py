import matplotlib as plt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from keras.preprocessing.image import load_img, img_to_array
from numpy.core.defchararray import array

path_train = './PetImages/'
path_train_cats = "./PetImages/Cat/"
path_train_dogs = "./PetImages/Dog/"

photos, labels = [], []

for file in listdir(path_train_dogs):
    print('Perros: ', file)

    # 0 for dogs, 1 for cats
    output = 0.0

    photo = load_img(path_train_dogs +  file, target_size=(48, 48))
    photo = img_to_array(photo)

    
    photos.append(photo)
    labels.append(output)

for file in listdir(path_train_cats):
    print('Gatos: ', file)

    # 0 for dogs, 1 for cats
    output = 1.0

    photo = load_img(path_train_cats +  file, target_size=(48, 48))
    photo = img_to_array(photo)

    photos.append(photo)
    labels.append(output)

photos = np.array(photos)
labels = np.array(labels)
print(photos.shape, labels.shape)

photos = photos / 255
print(photos[0])
print(photos[1])

np.save('dogs_vs_cats_photos.npy', photos)
np.save('dogs_vs_cats_labels.npy', labels)
    