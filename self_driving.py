#!/usr/bin/python3.7
import camera as cam
import joystick as js
import distance as ds
from time import sleep
from motor import Motor
from data_utils import *
from threading import Thread
from tensorflow.keras.models import load_model


def threaded(function):
    def wrapper(*args, **kwargs):
        Thread(target=function, args=args, kwargs=kwargs).start()
    return wrapper


@threaded
def distance():
    global dis

    while start:
        dis = np.round(ds.get_dis(), 1)
        sleep(0.5)


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
        distance()
        print(f'Self driving {"Started" if start else "Stopped"}')
        sleep(0.3)

    if start:
        img = cam.get_img(False, width=200, height=120)
        img = pre_process(img)
        img = np.array([img])

        steering = float(model.predict(img)) * steering_sensitivity
        # print(np.round(steering, 2))

        if dis > 20:
            motor.move(speed=max_speed, turn=steering, no_limit=True)
        else:
            motor.stop()

        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        motor.stop()


if __name__ == '__main__':
    motor = Motor(21, 20, 16, 26, 13, 19)
    max_speed = 0.2
    start = False
    dis = 0.0

    try:
        model = load_model(f'Models/{sys.argv[1]}.h5')
        steering_sensitivity = float(sys.argv[2])
    except (IndexError, ValueError):
        print(f'Give required arguments: Name of model and Steering sensitivity')
        sys.exit()

    js.init()
    ds.init()
    print('Ready for Self-Driving')

    # Add Traffic Sign Detection (haarcascade)

    while True:
        main()
