#!/usr/bin/python3.7
from training_utils import *
from tensorflow.keras.models import load_model


if __name__ == '__main__':
    steering_sensitivity = 1    # Maybe 0.7 ??
    model = load_model('model.h5')

    path = 'Training_Data'
    data = import_data_info(path=path, start_folder=0, end_folder=1)

    data = balance_data(data, display=False)
    images_path, steerings = load_data(path, data)

    index = random.randint(0, len(images_path) - 1)
    img = cv2.imread(images_path[index])
    img = pre_process(img)
    img = np.array([img])

    steering = float(model.predict(img))
    print(steering * steering_sensitivity)

    cv2.imshow('Test Image', cv2.imread(images_path[index]))
    cv2.waitKey(0)

    # path = 'Training_Data/IMG0/Image_1619189861989188.jpg'
    #
    # img = cv2.imread(path)
    # img = pre_process(img)
    # img = np.array([img])
    #
    # steering = float(model.predict(img))
    # print(steering * steering_sen)
    #
    # cv2.imshow('Test Image', cv2.imread(path))
    # cv2.waitKey(0)
