#!/usr/bin/python3.7
import camera as cam
import joystick as js
from time import sleep
from motor import Motor
from data_utils import *
from tensorflow.keras.models import load_model


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
        sleep(0.3)

    if start:
        img = cam.get_img(False, width=200, height=120)
        img = pre_process(img)
        img = np.array([img])

        steering = float(model.predict(img)) * steering_sensitivity
        # print(steering)

        motor.move(speed=max_speed, turn=steering, no_limit=True)
        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        motor.stop()


if __name__ == '__main__':
    motor = Motor(21, 20, 16, 26, 13, 19)
    max_speed = 0.2
    start = False

    try:
        model = load_model(f'Models/{sys.argv[1]}.h5')
        steering_sensitivity = float(sys.argv[2])
    except (IndexError, ValueError):
        print(f'Give required arguments: Name of model and Steering sensitivity')
        sys.exit()

    js.init()
    print('Ready for Self-Driving')

    # Add Traffic Sign Detection (haarcascade)

    while True:
        main()
