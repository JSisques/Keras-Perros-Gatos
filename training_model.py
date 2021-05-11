import matplotlib as plt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as kr

from keras.preprocessing.image import load_img, img_to_array
from numpy.core.defchararray import array

photos = np.load('dogs_vs_cats_photos.npy')
labels = np.load('dogs_vs_cats_labels.npy')

print(photos.shape, labels.shape)

photos = photos / 255
labels = labels / 255

print(photos[0])
print(labels[0])

#Definimos el modelo

model = kr.Sequential()

#Capa 1
model.add(kr.layers.Conv2D(32, activation='relu', kernel_size=(3,3),  input_shape=(150, 150, 3)))
model.add(kr.layers.MaxPooling2D(2,2))

#Capa 2
model.add(kr.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(kr.layers.MaxPooling2D(2,2))

#Capa 3
model.add(kr.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(kr.layers.MaxPooling2D(2,2))

#Capa 4
model.add(kr.layers.Flatten())
model.add(kr.layers.Dropout(0.5))

#Capa 5
model.add(kr.layers.Dense(256, activation='relu'))

#Capa 6 (Salida)
model.add(kr.layers.Dense(1, activation='sigmoid'))


model.compile(
    optimizer='sgd',
    metrics=['accuracy'],
    loss='binary_crossentropy'
)

print(model.summary())

model.fit(photos, labels, epochs=5)

model.save('model.h5')


