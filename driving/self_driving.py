#!/usr/bin/python3.7
from imports import *
from time import sleep, time
from threading import Thread
from modules.motor import Motor
from training.data_utils import *
from modules.distance import SRF05
from tensorflow.keras.models import load_model
from modules import camera as cam, joystick as js


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
        print('\nExit Self-Driving')
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

        if measure_fps:
            if len(tpf) <= 1000:
                tpf.append(time() - lft)
                lft = time()
            else:
                tpf.pop(0)
                avg_tpf = sum(tpf) / len(tpf)
                fps = 1 / avg_tpf
                print(fps)
                start = not start

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
    try:
        model = load_model(f'../data/models/{sys.argv[1]}.h5')
        steering_sensitivity = float(sys.argv[2])
        max_speed = float(sys.argv[3])
        save_images = True if int(sys.argv[4]) == 1 else False
        measure_fps = True if int(sys.argv[5]) == 1 else False
    except (IndexError, ValueError):
        print(f'Give required arguments: Name of model, Steering sensitivity, Max speed (0.00 - 1.00), Save images option (0 or 1) and Measure fps option (0 or 1)')
        print('python3.7 self_driving.py model 1 0.25 0 0')
        sys.exit()

    motor = Motor(21, 20, 16, 26, 13, 19)
    sensor = SRF05(trigger_pin=23, echo_pin=24)
    start = False
    distance = 0
    js.init()

    align = 7
    print(f'\nJoystick input:')
    print(f'{"START:":<{align}} Start/Stop\n{"SELECT:":<{align}} Quit\n')

    if save_images:
        path = '../data/test'
        count_folder = 0
        count_images = 0
        time_passed = True
        font = cv2.FONT_HERSHEY_SIMPLEX

        while os.path.exists(os.path.join(path, f'IMG{str(count_folder)}')):
            count_folder += 1

        path = path + "/IMG" + str(count_folder)
        os.makedirs(path)

    if measure_fps:
        lft = 0             # Last frame time
        tpf = []            # Time per frame

    print('Ready for Self-Driving\n')

    while True:
        main()
