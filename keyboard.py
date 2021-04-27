#!/usr/bin/python3.7
import sys
import pygame


def init():
    pygame.init()
    pygame.display.set_caption('keys')
    pygame.display.set_mode((200, 200))


def get_key(key_name):
    result = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    key_input = pygame.key.get_pressed()
    key_name = getattr(pygame, f'K_{key_name}')

    if key_input[key_name]:
        result = True

    pygame.display.update()

    return result


def main():
    if get_key('UP'):
        print('up')
    elif get_key('DOWN'):
        print('down')
    elif get_key('LEFT'):
        print('left')
    elif get_key('RIGHT'):
        print('right')


if __name__ == '__main__':
    init()

    while True:
        main()
