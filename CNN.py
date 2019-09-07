import random
import numpy as np
from data.handle_data import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class CNN:
    img_width, img_height = 150, 150
    epochs = 10
    batch_size = 64

    def __init__(self):
        self.trainX = self.testX = self.trainY = self.testY = None
        if K.image_data_format() == 'channels_first':
            input_shape = (3, self.img_width, self.img_height)
        else:
            input_shape = (self.img_width, self.img_height, 3)

        self.model = Sequential()
        self.model.add(Conv2D(32, (2, 2), input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def prepare_data(self, data_dir):

        data, labels = convert_dataset(data_dir)
        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(data,
                                                                              labels, test_size=0.25, random_state=random.randint(1, 100))

        # convert the labels from integers to vectors
        self.trainY = to_categorical(self.trainY, num_classes=2)
        self.testY = to_categorical(self.testY, num_classes=2)

    def train(self, data_dir="data"):
        self.prepare_data(data_dir)
        self.model.fit(self.trainX, self.trainY, validation_data=(self.testX, self.testY), epochs=self.epochs,
                       batch_size=self.batch_size)
        scores = self.model.evaluate(self.testX, self.testY, verbose=0)
        accuracy = scores[1] * 100
        print("Accuracy: %.2f%%" % accuracy)
        if accuracy > 85:
            self.model.save_weights("model.h5")

    def predict(self, file_to_predict):
        if os.path.exists("model.h5"):
            self.model.load_weights("model.h5")
        else:
            self.train()
        img = load_img(file_to_predict, target_size=(self.img_width, self.img_height))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = self.model.predict_classes(images, batch_size=10)
        cell_type = "Parasitized" if classes[0] == 1 else "Uninfected"
        return cell_type


if __name__ == '__main__':
    nn = CNN()
    prediction = nn.predict("test/C33P1thinF_IMG_20150619_114756a_cell_181.png")
    print(prediction)