#!/usr/bin/python3.7
from data_utils import *
from modules.motor import Motor
from tensorflow.keras.models import load_model
from modules import camera as cam, joystick as js


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
    print(np.round(steering, 2))

    cv2.waitKey(1)


if __name__ == '__main__':
    try:
        model = load_model(f'../data/models/{sys.argv[1]}.h5')
        steering_sensitivity = float(sys.argv[2])
        max_speed = float(sys.argv[3])
    except (IndexError, ValueError):
        print(f'Give required arguments: Name of model, Steering sensitivity and Max speed (0.00 - 1.00)')
        sys.exit()

    motor = Motor(21, 20, 16, 26, 13, 19)
    js.init()
    print('Ready to Check Model')

    while True:
        main()
