#!/usr/bin/python3.7
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
global trigger, echo


# Pin 20: Ground, Pin 18: Echo with Voltage Divider, Pin 16: Trigger, Pin 2: 5V
def init():
    global trigger, echo

    trigger = 23
    echo = 24

    GPIO.setup(trigger, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)


def get_dis():
    # Set Trigger to HIGH
    GPIO.output(trigger, True)

    # Set Trigger after 10us to LOW
    time.sleep(0.00001)
    GPIO.output(trigger, False)

    start_time = time.time()
    stop_time = time.time()

    # Save StartTime
    while GPIO.input(echo) == 0:
        start_time = time.time()

    # Save time of arrival
    while GPIO.input(echo) == 1:
        stop_time = time.time()

    # Time difference between start and arrival
    time_elapsed = stop_time - start_time

    # Multiply with the sonic speed (34300 cm/s) and divide by 2, because there and back
    distance = (time_elapsed * 34300) / 2

    return distance


if __name__ == '__main__':
    init()

    try:
        while True:
            print(f'Measured Distance = {get_dis():>4.1f} cm')
            # You should wait 100ms before the next trigger, even if the SRF05 detects a close object and the echo pulse is shorter.
            # This is to ensure the ultrasonic beep has faded away and will not cause a false echo on the next ranging.
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Measurement stopped by User")
