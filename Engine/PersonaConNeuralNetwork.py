import ntpath

import cv2
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import numpy as np
from keras.preprocessing import image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PersonaConNeuralNetwork:

    """"===================================================================================================
        ===================================================================================================

        PersonaConNeuralNetwork Class - Build, Manage and Predict using the Persona Convolution Neural Network

        Arguments:


            - type : two options : 1. Debug ( analysis process )
                                   2. Full ( analysis process + store analyzed products )

        Predict :

            The Persona Convolution Neural Network is a classification model that has an ability
            to classify 6 different classes of drawing's objects .

            The classes are:
                                - Cloud
                                - Door
                                - House
                                - Person
                                - Sun
                                - Tree

        ===================================================================================================
        ==================================================================================================="""

    def __init__(self, type):

        print()
        print("PersonaConNeuralNetwork : Building new instance !", end="\n\n")

        # self.image_path = image_path
        # self.image_filename = ntpath.basename(image_path)

        self.type = type




    def BuildPersonaCnnModel(self, numberOfClasses , width, height ):

        """ Initialize The Convolution Neural Network Model

            Arguments:
                * Number Of Classes : the number of classes the model has the ability to classify
                * width : width of the images of the training set / test set
                * height : height of the images of the training set / test set

            The model consist :
                - 5 Convolution Layers
                - 2 Fully Connected Layers                                                      """

        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(width, height, 1)))

        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(numberOfClasses, activation='softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's compile the model using RMSprop

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return self.model


    def create_image_generators(self):

        """ Generate batches of tensor image data with real-time data augmentation """

        self.train_datagen = ImageDataGenerator(rescale=1. / 255,
                                                rotation_range=20.,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                shear_range=0.1,
                                                zoom_range=0.2,
                                                horizontal_flip=True)

        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        return self.train_datagen, self.test_datagen


    def create_image_datasets(self , training_set_dir, test_set_dir , width=64, height=64):

        """ generates batches of augmented data from a directory """

        self.training_set = self.train_datagen.flow_from_directory(training_set_dir,
                                                         target_size=(width, height),
                                                         color_mode='grayscale',
                                                         batch_size=32,
                                                         class_mode='categorical')

        self.test_set = self.test_datagen.flow_from_directory(test_set_dir,
                                                    target_size=(width, height),
                                                    color_mode='grayscale',
                                                    batch_size=32,
                                                    class_mode='categorical')

        return self.training_set, self.test_set


    def load_model_persona(self , pathToModel):

        self.model = load_model(pathToModel)

        return self.model

    def save_model_persona(self, pathToModel ):

        self.model.save(pathToModel)

    def fit_model(self , num_of_epoch=1):

        self.model.fit_generator(self.training_set,
        steps_per_epoch = 20,
        epochs = num_of_epoch,
        validation_data = self.test_set,
        validation_steps = 120)

    def predict_persona(self, image_to_classification):

        if self.model is not None:

            result = self.model.predict_classes(image_to_classification)  # training_set.class_indices

            if result[0] == 0:
                return "Cloud"
            elif result[0] == 1:
                return "Door"
            elif result[0] == 2:
                return "House"
            elif result[0] == 3:
                return "Person"
            elif result[0] == 4:
                return "Sun"
            elif result[0] == 5:
                return "Tree"
            else:
                return "Unknown"

        else:

            return "Error: Model isn't define yet ."

    def load_image_for_predict(self, path, size):

        # print()

        test_image = image.load_img(path, target_size=(size[0], size[1]), grayscale=True)

        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis=0)

        print("image loaded successfully and ready to classify , shape = " , test_image.shape)

        return test_image













