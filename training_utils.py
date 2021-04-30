#!/usr/bin/python3.7
import tensorflow as tf
import matplotlib.image as mpimg
from data_utils import *
from imgaug import augmenters as iaa
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model

config = tf.compat.v1.ConfigProto()             # Error: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERRO
config.gpu_options.allow_growth = True          # The process grow the memory usage as it is needed by the process
sess = tf.compat.v1.Session(config=config)


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


# Create model, by nvidia: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
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
            index = np.random.randint(len(images_path))

            if train_flag:
                img, steering = augment_image(images_path[index], steering_list[index])
            else:
                img = mpimg.imread(images_path[index])
                steering = steering_list[index]

            img = pre_process(img)
            img_batch.append(img)
            steering_batch.append(steering)

        yield np.asarray(img_batch), np.asarray(steering_batch)


def main():
    path = 'Training_Data'
    data = import_data_info(path=path, start_folder=0, end_folder=1)
    images_path, steerings = load_data(path, data)

    index = np.random.randint(len(data))
    image_path = images_path[index]
    steering = steerings[index]

    # Check augmentation image function
    check_augment_image = True

    if check_augment_image:
        fig = plt.figure(figsize=(8, 6))
        fig.canvas.set_window_title('Augmentation image')
        sub = fig.add_subplot(1, 2, 1)
        sub.set_title(f'Steering before augmentation: {steering}')
        image = mpimg.imread(image_path)
        plt.imshow(image)

        img, st = augment_image(image_path, steering)
        sub = fig.add_subplot(1, 2, 2)
        sub.set_title(f'Steering after augmentation: {st}')
        plt.imshow(img)
        plt.show()

    # Check preprocess image function
    check_pre_process = True

    if check_pre_process:
        fig = plt.figure(figsize=(8, 6))
        fig.canvas.set_window_title('Preprocess image')
        sub = fig.add_subplot(1, 2, 1)
        sub.set_title(f'Before preprocess')
        image = mpimg.imread(image_path)
        plt.imshow(image)

        img = pre_process(mpimg.imread(image_path))
        sub = fig.add_subplot(1, 2, 2)
        sub.set_title(f'After preprocess')
        plt.imshow(img)
        plt.show()

    # Check model with saved images
    check_model = True

    if check_model:
        model = load_model('Models/model.h5')
        steering_sensitivity = 1
        test_images = 10

        for i in range(test_images):
            index = np.random.randint(len(images_path))
            img = cv2.imread(images_path[index])
            img = pre_process(img)
            img = np.array([img])

            steering = -float(model.predict(img)) * steering_sensitivity
            print(steering)

            cv2.imshow('Test Image', cv2.imread(images_path[index]))
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
