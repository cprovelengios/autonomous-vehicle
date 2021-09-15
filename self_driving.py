#!/usr/bin/python3.7
import camera as cam
import joystick as js
from time import sleep, time
from motor import Motor
from data_utils import *
from distance import SRF05
from threading import Thread
from tensorflow.keras.models import load_model


def measure():
    global distance

    while start:
        distance = sensor.measure()


def thread_sleep():
    global time_passed

    while start:
        if not time_passed:
            sleep(1)
            time_passed = True


def main():
    global start, time_passed, count_images, lft

    js_values = js.get_js()

    if js_values['select'] == 1:
        cv2.destroyAllWindows()
        start = False
        motor.stop()
        sys.exit()
    elif js_values['start'] == 1:
        start = not start
        Thread(target=measure).start()

        if save_images:
            Thread(target=thread_sleep).start()

        print(f'Self driving {"Started" if start else "Stopped"}')
        sleep(0.3)

    if start:
        img = image = cam.get_img(False, width=200, height=120)
        img = pre_process(img)
        img = np.array([img])

        steering = float(model.predict(img)) * steering_sensitivity
        # print(np.round(steering, 2))

        if distance is not None and distance > 15:
            motor.move(speed=max_speed, turn=steering, no_limit=True)
        else:
            motor.stop()

        # if len(tpf) <= 1000:
        #     tpf.append(time() - lft)
        #     lft = time()
        # else:
        #     tpf.pop(0)
        #     avg_tpf = sum(tpf) / len(tpf)
        #     fps = 1 / avg_tpf
        #     print(fps)
        #     start = not start

        if save_images and time_passed:
            time_passed = False

            cv2.putText(image, str(np.round(steering, 2)), (0, 17), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

            if count_images < 10:
                cv2.putText(image, str(count_images), (184, 17), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, str(count_images), (170, 17), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

            file_name = os.path.join(path, f'Image_{count_images}.jpg')
            cv2.imwrite(file_name, image)
            count_images += 1

        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        motor.stop()


if __name__ == '__main__':
    motor = Motor(21, 20, 16, 26, 13, 19)
    sensor = SRF05(trigger_pin=23, echo_pin=24)
    start = False
    distance = 0

    lft = 0             # Last frame time
    tpf = []            # Time per frame

    count_images = 0
    time_passed = True
    font = cv2.FONT_HERSHEY_SIMPLEX
    path = os.path.join(os.getcwd(), 'Test')

    try:
        model = load_model(f'Models/{sys.argv[1]}.h5')
        steering_sensitivity = float(sys.argv[2])
        max_speed = float(sys.argv[3])
        save_images = True if int(sys.argv[4]) == 1 else False
    except (IndexError, ValueError):
        print(f'Give required arguments: Name of mode, Steering sensitivity, Max speed(0.00 - 1.00) and Save images option(0 or 1)')
        sys.exit()

    js.init()
    print('Ready for Self-Driving')

    while True:
        main()
