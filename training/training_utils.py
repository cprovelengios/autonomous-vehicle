#!/usr/bin/python3.7
import tensorflow as tf
from data_utils import *
from imgaug import augmenters as iaa
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model

config = tf.compat.v1.ConfigProto()             # Error: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERRO
config.gpu_options.allow_growth = True          # The process grows the memory usage as it is needed by the process
sess = tf.compat.v1.Session(config=config)


# Data augmentation
def augment_image(img_path, steering):
    img = cv2.imread(img_path)

    # Translational augmentation moves the image along the x and y direction
    # if np.random.rand() < 0.5:
    #     pan = iaa.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, cval=155)
    #     img = pan.augment_image(img)

    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.6, 2))
        img = brightness.augment_image(img)

    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


# Create model, by Nvidia: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
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
                img = cv2.imread(images_path[index])
                steering = steering_list[index]

            img = pre_process(img)
            img_batch.append(img)
            steering_batch.append(steering)

        yield np.asarray(img_batch), np.asarray(steering_batch)


# Check augmentation image function
def check_augmentation(images_path, steerings):
    for i in range(check_augment):
        index = np.random.randint(len(images_path))

        image = cv2.imread(images_path[index])
        image = pre_process(image)
        image = cv2.resize(image, (640, 360))

        img, st = augment_image(images_path[index], steerings[index])
        img = pre_process(img)
        img = cv2.resize(img, (640, 360))

        img_concate_hori = np.concatenate((image, img), axis=0)             # Horizontally: axis=1, Vertically: axis=0
        cv2.imshow(f'Steerings: Before augmentation {steerings[index]}  -  After augmentation {st}', img_concate_hori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Check Model Predictions with saved images
def check_model(images_path, steerings):
    model = load_model(f'../data/models/{name_model}.h5')
    index = 0
    length = len(images_path)

    while True:
        img = cv2.imread(images_path[index])
        img = test_img = pre_process(img)
        img = np.array([img])

        steering = float(model.predict(img)) * steering_sensitivity
        print(f'Image: {index:>4}  -  Prediction: {np.round(steering, 2):>5}  -  True: {steerings[index]:>5}')

        test_img = cv2.resize(test_img, (640, 360))
        cv2.imshow('Test Image', test_img)
        key = cv2.waitKey(0)

        if key == 83:                       # ->
            index = (index + 1) % length
        elif key == 81:                     # <-
            index = (index - 1) % length
        elif key == 27:                     # esc
            break


def main():
    path = '../data/training_data'
    data = import_data_info(path=path, start_folder=folders[0], end_folder=folders[1])
    images_path, steerings = load_data(data)

    # Check augmentation image function
    if check_augment:
        check_augmentation(images_path, steerings)

    # Check Model Predictions with saved images
    if name_model:
        check_model(images_path, steerings)


if __name__ == '__main__':
    try:
        folders = list(map(int, sys.argv[1].split('-')))
        check_augment = int(sys.argv[2])

        try:
            name_model = sys.argv[3]
            steering_sensitivity = float(sys.argv[4])
        except IndexError:
            name_model = ''
            steering_sensitivity = 0
    except (IndexError, ValueError):
        print(f'Give required arguments: Start folder-End folder (0-0), Check augment (0 or Number), Name of model and Steering sensitivity if want check model')
        sys.exit()

    main()
