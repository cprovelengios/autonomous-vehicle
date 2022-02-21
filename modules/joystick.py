#!/usr/bin/python3.7
import pygame
from time import sleep

global controller, buttons, axiss


# Left: (axis1: -1 1 <>, axis2: -1 1 ^v), Right: (axis3: -1 1 <>, axis4: -1 1 ^v)
def init():
    global controller, buttons, axiss

    pygame.init()
    controller = pygame.joystick.Joystick(0)
    controller.init()

    buttons = {'x': 0, 'o': 0, 't': 0, 's': 0,
               'L1': 0, 'R1': 0, 'L2': 0, 'R2': 0,
               'select': 0, 'start': 0,
               'axis1': 0., 'axis2': 0., 'axis3': 0., 'axis4': 0.}
    axiss = [0., 0., 0., 0., 0., 0.]


def get_js(name=''):
    for event in pygame.event.get():                                # Retrieve any events
        if event.type == pygame.JOYAXISMOTION:                      # Analog Sticks
            axiss[event.axis] = round(event.value, 2)
        elif event.type == pygame.JOYBUTTONDOWN:                    # When button pressed
            # print(event.dict, event.joy, event.button, 'PRESSED')
            for x, (key, val) in enumerate(buttons.items()):
                if x < 10:
                    if controller.get_button(x):
                        buttons[key] = 1
        elif event.type == pygame.JOYBUTTONUP:                      # When button released
            # print(event.dict, event.joy, event.button, 'released')
            for x, (key, val) in enumerate(buttons.items()):
                if x < 10:
                    if event.button == x:
                        buttons[key] = 0

    # Put axis values at buttons dictionary.
    buttons['axis1'], buttons['axis2'], buttons['axis3'], buttons['axis4'] = [axiss[0], axiss[1], axiss[3], axiss[4]]

    if name == '':
        return buttons
    else:
        return buttons[name]


def main():
    print(get_js())             # To get all values
    # print(get_js('x'))          # To get a single value
    sleep(0.05)


if __name__ == '__main__':
    init()

    while True:
        main()
