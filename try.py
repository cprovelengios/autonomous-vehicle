#!/usr/bin/python3.7
import threading
from time import sleep

# Distance sensor will run on other thread, because need sleep


def threaded(func):
    def wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs).start()
    return wrapper


@threaded
def function():
    for i in range(10):
        print(i)
        sleep(0.5)


function()

for j in range(10):
    print(j)
    sleep(1)
