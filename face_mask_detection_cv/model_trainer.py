import cv2
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from keras.models import load_model

def load_data(path=None):
    images = []
    target = []

    class_label = -1
    if path is not None:

        for idx, each_folder in tqdm(enumerate(os.listdir(path))):
            all_files = os.listdir(path + each_folder)
            class_label += 1
            for img_idx in tqdm(range(len(all_files))):
                each_img = all_files[img_idx]
                img_path = path + each_folder + '\\' + each_img
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

                images.append(img)
                target.append(class_label)

        return images, target

    else:
        print('Null path')


def train_model(data, target):
    X = np.array(data)
    Y = np.array(target)

    xtr, xte, ytr, yte = train_test_split(X, Y, test_size=0.2)
    xtr = np.array(xtr)
    ytr = np.array(ytr)
    xte = np.array(xte)
    yte = np.array(yte)

    xtr = xtr / 255.0
    xte = xte / 255.0

    ytr = to_categorical(ytr)
    yte = to_categorical(yte)

    image_size = xtr[0].shape[0]
    opt_size = ytr[0].shape[0]

    xtr = xtr.reshape(xtr.shape[0], 224, 224, 3)
    xte = xte.reshape(xte.shape[0], 224, 224, 3)

    model = Sequential()

    model.add(Conv2D(16, input_shape=(image_size, image_size, 3), kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=1))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=1))

    model.add(Conv2D(256, kernel_size=1, activation='tanh'))  ##new layer

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(opt_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtr, ytr, epochs=5, validation_data=(xte, yte))

    return model


def save_model(model):
    model.save('face_detector/mask_detector2.h5')


if __name__ == "__main__":
    dataset_path = 'dataset\\'
    data, target = load_data(dataset_path)
    model = train_model(data, target)
    save_model(model)
