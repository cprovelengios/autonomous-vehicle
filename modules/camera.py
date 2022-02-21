#!/usr/bin/python3.7
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)     # frame width
cap.set(4, 360)     # frame height


def get_img(display=False, width=640, height=360):
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
