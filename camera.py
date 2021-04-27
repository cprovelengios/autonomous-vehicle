#!/usr/bin/python3.7
import cv2

# Camera Resolution: 1920 * 1080 (which is 1080p) creates an image that is 1920 pixels wide and 1080 pixels tall.
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)


def get_img(display=False, width=640, height=480):
    ret, img = cap.read()
    img = cv2.resize(img, (width, height))

    if display:
        cv2.imshow('IMG', img)

    return img


def main():
    get_img(True)
    cv2.waitKey(1)


if __name__ == '__main__':
    while True:
        main()
