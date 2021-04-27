#!/usr/bin/python3.7
import os
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from tensorflow.keras.layers import *

config = tf.compat.v1.ConfigProto()             # Error: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERRO
config.gpu_options.allow_growth = True          # The process grow the memory usage as it is needed by the process
sess = tf.compat.v1.Session(config=config)


# Import data info
def import_data_info(*, path, start_folder, end_folder):
    columns = ['Image', 'Steering']
    data = pd.DataFrame()

    for x in range(start_folder, end_folder + 1):
        data_new = pd.read_csv(os.path.join(path, f'log_{x}.csv'), names=columns)
        data_new['Image'] = data_new['Image'].str.split('/', 6).str[-1]
        print(f'Folder {x}: {data_new.shape[0]} Images', end=', ')
        data = data.append(data_new, True)  # True need to continue row index from last append

    print(f'\nTotal: {end_folder - start_folder + 1} Folder(s), {data.shape[0]} Images Imported\n')

    return data


# Visualize and balance data
def balance_data(data, display=True):
    number_of_bins = 201    # It defines the number of equal-width bins in the given range. Steering take values from -1 to 1, so (2 / 0.01) + 1 = 201, + 1 for 0 value
    samples_per_bin = 300   # It defines how many samples it will keep after balance for every bin
    remove_index_list = []

    hist, bins = np.histogram(data['Steering'], number_of_bins)
    center = np.round(np.arange(-1, 1.01, 0.01), 2)
    bins = np.round(bins, 2)

    if display:
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samples_per_bin, samples_per_bin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()

    flag = False    # This flag and following check need because bins have 2 times value 0

    for j in range(number_of_bins + 1):
        if bins[j] == 0 and not flag:
            flag = True
            continue

        bin_data_list = []

        for i in range(len(data['Steering'])):
            if data['Steering'][i] == bins[j]:
                bin_data_list.append(i)

        bin_data_list = shuffle(bin_data_list)
        bin_data_list = bin_data_list[samples_per_bin:]
        remove_index_list.extend(bin_data_list)

    data.drop(data.index[remove_index_list], inplace=True)
    print(f'Removed Images: {len(remove_index_list)}\nRemaining Images: {len(data)}')

    if display:
        hist, bins = np.histogram(data['Steering'], number_of_bins)
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samples_per_bin, samples_per_bin))
        plt.title('Balanced Data')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()

    return data


# Convert data frame to list
def load_data(path, data):
    images_path = []
    steering = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        images_path.append(os.path.join(path, indexed_data[0]))
        steering.append(float(indexed_data[1]))

    images_path = np.asarray(images_path)
    steering = np.asarray(steering)

    return images_path, steering


# Data augmentation
def augment_image(img_path, steering):
    img = mpimg.imread(img_path)

    # Translational augmentation moves the image along the x and y direction
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)

    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 2))
        img = brightness.augment_image(img)

    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


# Preprocess image for neural network, by nvidia ??? Must same on predict!!!
def pre_process(img):
    img = img[20:, :, :]        # Crop image
    img = img / 255             # Normalization

    return img


# # Preprocess image for neural network, by nvidia ??? Must same on predict!!!
# def pre_process(img):
#     img = img[54:120, :, :]                     # Crop image
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert image to YUV colorspace, not sure about this!!!
#     img = cv2.GaussianBlur(img,  (3, 3), 0)
#     img = cv2.resize(img, (200, 66))
#     img = img / 255                             # Normalization
#
#     return img


# Create model, by nvidia ??? Check bookmark!!!
def create_model():
    model = tf.keras.models.Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(tf.keras.optimizers.Adam(lr=0.0001), loss='mse')

    return model


# Data generator
def data_gen(images_path, steering_list, batch_size, train_flag):
    while True:
        img_batch = []
        steering_batch = []

        for i in range(batch_size):
            # index = np.random.randint(len(images_path))   # For some reason this works worst
            index = random.randint(0, len(images_path) - 1)

            if train_flag:
                img, steering = augment_image(images_path[index], steering_list[index])
            else:
                img = mpimg.imread(images_path[index])
                steering = steering_list[index]

            img = pre_process(img)
            img_batch.append(img)
            steering_batch.append(steering)

        yield np.asarray(img_batch), np.asarray(steering_batch)
