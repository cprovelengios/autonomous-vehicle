#!/usr/bin/python3.7
from data_utils import *
from training_utils import *
from tensorflow.keras.models import load_model


if __name__ == '__main__':
    steering_sensitivity = 1    # Maybe 0.7 ??
    model = load_model('Models/model_v0.h5')

    path = 'Training_Data'
    data = import_data_info(path=path, start_folder=0, end_folder=1)

    images_path, steerings = load_data(path, data)
    test_images = 20

    for i in range(test_images):
        index = np.random.randint(len(images_path))
        img = cv2.imread(images_path[index])
        img = pre_process(img)
        img = np.array([img])

        steering = float(model.predict(img))
        print(steering * steering_sensitivity)

        cv2.imshow('Test Image', cv2.imread(images_path[index]))
        cv2.waitKey(0)
