#!/usr/bin/python3.7
import sys
import cv2
import numpy as np
import camera as cam
import joystick as js
from time import sleep
from motor import Motor
from tensorflow.keras.models import load_model


# Preprocess image for neural network, MUST be same on Training and Predict!
def pre_process(img):
    img = img[40:, :, :]                        # Crop image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert image to YUV colorspace
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = img / 255                             # Normalization

    return img


def main():
    global start

    js_values = js.get_js()

    if js_values['select'] == 1:
        cv2.destroyAllWindows()
        motor.stop()
        sys.exit()
    elif js_values['start'] == 1:
        start = not start
        print(f'Self driving {"Started" if start else "Stopped"}')
        sleep(0.5)

    if start:
        img = cam.get_img(True, width=200, height=106)
        img = pre_process(img)
        img = np.array([img])

        steering = -float(model.predict(img)) * steering_sensitivity
        print(steering)

        motor.move(speed=max_speed, turn=steering)
        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        motor.stop()


if __name__ == '__main__':
    motor = Motor(21, 20, 16, 26, 13, 19)
    max_speed = 0.25
    steering_sensitivity = 1
    start = False

    js.init()
    model = load_model('Models/model_yuv_tape_29_04_2021-14:00:50.h5')
    # Add Traffic Sign Detection (haarcascade)

    while True:
        main()
