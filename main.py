import tensorflow.keras as kr
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import load_model, save_model
from keras.preprocessing import image
from skimage import io

from os import listdir
from os.path import isfile, join


img_width = 150
img_height = 150

train_data_directory = './Training/'
validation_data_directory = './Validation/'
test_data_directory = './Test/'

train_samples = 1500
validation_samples = 200
epochs = 10
batch_size = 10
input_shape = (img_width, img_height, 3)


def main():
    '''
    model = create_model()

    train_generator = create_train_datagen()
    imgs, labels = next(train_generator)
    print(imgs.shape, labels.shape)
    #show_image(imgs[0])

    image_batch, label_batch = train_generator.next()
    #show_classes(image_batch, label_batch)

    validation_generator = create_train_datagen()
    imgs, labels = next(validation_generator)
    print(imgs.shape, labels.shape)
    #show_image(imgs[0])

    image_batch, label_batch = validation_generator.next()
    #show_classes(image_batch, label_batch)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples
    )


    show_net_stats(history)
    save_model(model)
    '''

    model = load_custom_model()

    predict_images(model)



def create_model():
    model = kr.Sequential()

    # Capa 1
    model.add(kr.layers.Conv2D(32, kernel_size=(3, 3),
              input_shape=input_shape, activation='relu'))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2)))

    # Capa 2
    model.add(kr.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2)))

    # Capa 3
    model.add(kr.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2)))

    # Capa 4
    model.add(kr.layers.Flatten())
    model.add(kr.layers.Dropout(0.5))

    # Capa 5
    model.add(kr.layers.Dense(64, activation='relu'))

    # Capa 6
    model.add(kr.layers.Dense(1, activation='sigmoid'))

    # Compilacion
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Resumen de la red creada
    model.summary()

    return model


def create_train_datagen():
    # Indicamos que es lo que va a hacer con la imagen a la hora de entrenar
    train_datagen = ImageDataGenerator(
        # Reescalar los pixeles entre 0 y 1
        rescale=1. / 255,
        shear_range=0.2,
        # Hacer un zoom en la imagen
        zoom_range=0.2,
        # Hacer un flip horizontal
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    print(train_generator)
    print(train_generator.class_indices)

    return train_generator


def create_validation_datagen():
    # Indicamos que es lo que va a hacer con la imagen a la hora de entrenar
    validation_datagen = ImageDataGenerator(
        # Reescalar los pixeles entre 0 y 1
        rescale=1. / 255,
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    return validation_generator


def show_image(img):
    io.imshow(img)
    io.show()


def show_classes(image_batch, label_batch):
    for i in range(0, len(image_batch)):
        image = image_batch[i]
        print(label_batch[i])
        show_image(image)

def show_net_stats(history):
    #Listar todos los datos del modelo
    print(history.history.keys())

    #Graficar los aciertos
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #Graficar las perdidas
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def save_model(model):
    model.save('model.h5')

def load_custom_model():
    return load_model('model1.h5')

def predict_images(model):
    counter = 0
    dog_counter = 0
    cat_counter = 0

    for file in listdir(test_data_directory):
        try:
            img = image.load_img(test_data_directory + file, target_size=(img_width, img_height))
            x = image.img_to_array(img)
            x = np.expand_dims(x, 0)

            images = np.stack([img])
            classes = model.predict_classes(images, batch_size=10)

            print('Classes: ', classes)

            counter += 1

            if classes == 0:
                print(file + ": cat")
                cat_counter += 1
                os.rename(test_data_directory + file, "./Cats/" + file)
            else:
                print(file + ": dog")
                dog_counter += 1
                os.rename(test_data_directory + file, "./Dogs/" + file)
        except:
            print('Error')
    
    print('Numero de fotos analizadas: ', counter)
    print('Numero de perros: ', dog_counter)
    print('Numero de gatos: ', cat_counter)



main()
