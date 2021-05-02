#!/usr/bin/python3.7
import os
import sys
import cv2
import pandas as pd
import camera as cam
import joystick as js
from time import sleep
from motor import Motor
from datetime import datetime


# Save data, image and steering value
def save_data(image, steering_value):
    timestamp = str(datetime.timestamp(datetime.now())).replace('.', '')

    file_name = os.path.join(new_path, f'Image_{timestamp}.jpg')
    cv2.imwrite(file_name, image)

    img_list.append(file_name)
    steering_list.append(steering_value)


# Save Log file when session ends
def save_log():
    raw_data = {'Image': img_list, 'Steering': steering_list}
    df = pd.DataFrame(raw_data)
    df.to_csv(os.path.join(path, f'log_{str(count_folder)}.csv'), index=False, header=False)    # index: row name, header: column name
    print(f'Log Saved, total images: {len(img_list)}')


def main():
    global record

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
    elif js_values['start'] == 1:
        print(f'Recording {"Started" if record == 0 else "Stopped"}')
        record += 1
        sleep(0.3)      # sleep until user release the button

    if record == 1:
        img = cam.get_img(False, width=240, height=120)
        save_data(img, steering)
    elif record == 2:
        save_log()
        cv2.destroyAllWindows()
        record = 0
        print()

    motor.move(speed=throttle, turn=steering)
    cv2.waitKey(1)


if __name__ == '__main__':
    count_folder = 0
    img_list = []
    steering_list = []

    # Create new folder for current session
    path = os.path.join(os.getcwd(), 'Training_Data')

    while os.path.exists(os.path.join(path, f'IMG{str(count_folder)}')):
        count_folder += 1

    new_path = path + "/IMG" + str(count_folder)
    os.makedirs(new_path)

    motor = Motor(21, 20, 16, 26, 13, 19)
    max_speed = 0.25
    record = 0

    js.init()

    while True:
        main()
