#!/usr/bin/python3.7
import cv2
import numpy as np
import camera as cam
from motor import Motor
from tensorflow.keras.models import load_model


# Preprocess image for neural network, MUST be same on Training and Predict!
def pre_process(img):
    img = img[20:, :, :]        # Crop image
    img = img / 255             # Normalization

    return img


def main():
    # img = cam.get_img(False, width=200, height=86)
    img = cam.get_img(True, width=200, height=86)
    img = np.asarray(img)   # Need this ?
    img = pre_process(img)
    img = np.array([img])

    steering = float(model.predict(img))
    print(steering * steering_sensitivity)  # When done comment this

    motor.move(speed=max_speed, turn=steering * steering_sensitivity)
    cv2.waitKey(1)


if __name__ == '__main__':
    motor = Motor(21, 20, 16, 26, 13, 19)
    max_speed = 0.25
    steering_sensitivity = 1    # Maybe 0.7 ?

    model = load_model('model.h5')
    # Add Traffic Sign Detection, just a haarcascade.

    while True:
        main()
