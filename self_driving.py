#!/usr/bin/python3.7
import camera as cam
import joystick as js
from time import sleep
from motor import Motor
from data_utils import *
from distance import SRF05
from threading import Thread
from tensorflow.keras.models import load_model


def measure():
    global distance

    while start:
        distance = sensor.measure()


def main():
    global start

    js_values = js.get_js()

    if js_values['select'] == 1:
        cv2.destroyAllWindows()
        start = False
        motor.stop()
        sys.exit()
    elif js_values['start'] == 1:
        start = not start
        Thread(target=measure).start()
        print(f'Self driving {"Started" if start else "Stopped"}')
        sleep(0.3)

    if start:
        img = cam.get_img(False, width=200, height=120)
        img = pre_process(img)
        img = np.array([img])

        steering = float(model.predict(img)) * steering_sensitivity
        # print(np.round(steering, 2))

        if distance is not None and distance > 15:
            motor.move(speed=max_speed, turn=steering, no_limit=True)
        else:
            motor.stop()

        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        motor.stop()


if __name__ == '__main__':
    motor = Motor(21, 20, 16, 26, 13, 19)
    sensor = SRF05(trigger_pin=23, echo_pin=24)
    start = False
    distance = 0

    try:
        model = load_model(f'Models/{sys.argv[1]}.h5')
        steering_sensitivity = float(sys.argv[2])
        max_speed = float(sys.argv[3])
    except (IndexError, ValueError):
        print(f'Give required arguments: Name of mode, Steering sensitivity and Max speed(0.00 - 1.00)')
        sys.exit()

    js.init()
    print('Ready for Self-Driving')

    while True:
        main()
