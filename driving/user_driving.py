#!/usr/bin/python3.7
from imports import *
from time import sleep
from modules.motor import Motor
from training.data_utils import *
from modules import camera as cam, joystick as js, keyboard as kb


def main():
    global speed_option, joystick_option, camera_option, camera

    if movement == 'joystick':
        js_values = js.get_js()

        if joystick_mode[joystick_option] == 'buttons':
            steering = js_values['axis3']
            throttle = 0

            if js_values['R1']:
                throttle = max_speed[speed_option]
            elif js_values['L1']:
                throttle = -max_speed[speed_option]
            elif js_values['R2']:
                speed_option = (speed_option + 1) % 4
                print(f'Max speed: {max_speed[speed_option] * 100:>3.0f}%')
                sleep(0.3)
            elif js_values['L2']:
                speed_option = (speed_option - 1) % 4
                print(f'Max speed: {max_speed[speed_option] * 100:>3.0f}%')
                sleep(0.3)
        else:
            steering = js_values['axis3']
            throttle = -js_values['axis2']

        if js_values['x'] == 1:
            joystick_option = (joystick_option + 1) % 2
            print(f'Joystick mode {joystick_mode[joystick_option]}')
            sleep(0.3)
        elif js_values['s'] == 1:
            camera_option = (camera_option + 1) % 2
            sleep(0.3)
        elif js_values['select'] == 1:
            print('\nExit User-Driving')
            cv2.destroyAllWindows()
            motor.stop()
            sys.exit()
        elif js_values['start'] == 1:
            camera = not camera
            sleep(0.3)
    else:
        steering = 0
        throttle = 0

        if kb.get_key('UP'):
            throttle = max_speed[speed_option]
        elif kb.get_key('DOWN'):
            throttle = -max_speed[speed_option]

        if kb.get_key('LEFT'):
            steering = -1
        elif kb.get_key('RIGHT'):
            steering = 1

        if kb.get_key('d'):
            speed_option = (speed_option + 1) % 4
            print(f'Max speed: {max_speed[speed_option] * 100:>3.0f}%')
            sleep(0.3)
        elif kb.get_key('a'):
            speed_option = (speed_option - 1) % 4
            print(f'Max speed: {max_speed[speed_option] * 100:>3.0f}%')
            sleep(0.3)
        elif kb.get_key('w'):
            camera_option = (camera_option + 1) % 2
            sleep(0.3)
        elif kb.get_key('s'):
            camera = not camera
            sleep(0.3)
        elif kb.get_key('ESCAPE'):
            print('\nExit User-Driving')
            cv2.destroyAllWindows()
            motor.stop()
            sys.exit()

    motor.move(speed=throttle, turn=steering)

    if camera:
        if camera_mode[camera_option] == 'normal':
            cam.get_img(True)
        else:
            img = cam.get_img(False, 200, 120)
            img = pre_process(img)
            img = cv2.resize(img, (640, 360))
            cv2.imshow('IMG', img)
    else:
        cv2.destroyAllWindows()

    cv2.waitKey(1)


if __name__ == '__main__':
    try:
        get_movement = ['joystick', 'keyboard']
        joystick_mode = ['buttons', 'analog']
        joystick_option = 0

        movement = get_movement[int(sys.argv[1])]

        if movement == 'joystick':
            js.init()

            align = 7
            print(f'\nJoystick mode {joystick_mode[joystick_option]} input:')
            print(f'{"R1:":<{align}} Throttle\n{"L1:":<{align}} Reverse\n{"R2:":<{align}} Speed++')
            print(f'{"L2:":<{align}} Speed--\n{"x:":<{align}} Input Mode\n{"START:":<{align}} Camera On/Off')
            print(f'{"s:":<{align}} Camera Mode\n{"SELECT:":<{align}} Quit\n')
        else:
            kb.init()

            align = 6
            print(f'\nKeyboard input:')
            print(f'{"UP:":<{align}} Throttle\n{"DOWN:":<{align}} Reverse\n{"LEFT:":<{align}} Left')
            print(f'{"RIGHT:":<{align}} Right\n{"d:":<{align}} Speed++\n{"a:":<{align}} Speed--')
            print(f'{"s:":<{align}} Camera On/Off\n{"w:":<{align}} Camera Mode\n{"ESC:":<{align}} Quit\n')
    except (IndexError, ValueError):
        print(f'\nGive required argument: Input mode (0 for joystick - 1 for keyboard)')
        print('python3.7 user_driving.py 0')
        sys.exit()

    motor = Motor(21, 20, 16, 26, 13, 19)
    max_speed = [0.25, 0.5, 0.75, 1]
    speed_option = 0

    camera_mode = ['normal', 'pre_process']
    camera_option = 0
    camera = False

    print('Ready for User-Driving\n')

    while True:
        main()
