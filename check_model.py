#!/usr/bin/python3.7
import camera as cam
import joystick as js
from motor import Motor
from data_utils import *
from tensorflow.keras.models import load_model


def main():
    js_values = js.get_js()

    steering = js_values['axis3']
    throttle = 0

    if js_values['R1']:
        throttle = max_speed
    elif js_values['L1']:
        throttle = -max_speed
    elif js_values['select'] == 1:
        cv2.destroyAllWindows()
        motor.stop()
        sys.exit()

    motor.move(speed=throttle, turn=steering)

    img = cam.get_img(True, width=200, height=120)
    img = pre_process(img)
    img = np.array([img])

    steering = float(model.predict(img)) * steering_sensitivity
    print(steering)

    cv2.waitKey(1)


if __name__ == '__main__':
    motor = Motor(21, 20, 16, 26, 13, 19)
    max_speed = 0.5

    try:
        model = load_model(f'Models/{sys.argv[1]}.h5')
        steering_sensitivity = float(sys.argv[2])
    except (IndexError, ValueError):
        print(f'Give required arguments: Name of model and Steering sensitivity')
        sys.exit()

    js.init()

    while True:
        main()
