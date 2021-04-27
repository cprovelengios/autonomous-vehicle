#!/usr/bin/python3.7
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


# Pin 39: Ground, Pin 37: Red, Pin 40: Red(Tape), a: right motors, b: left motors
class Motor:
    def __init__(self, en_a, in_a1, in_a2, en_b, in_b1, in_b2):
        self.en_a = en_a
        self.in_a1 = in_a1
        self.in_a2 = in_a2
        self.en_b = en_b
        self.in_b1 = in_b1
        self.in_b2 = in_b2

        GPIO.setup(self.en_a, GPIO.OUT)
        GPIO.setup(self.in_a1, GPIO.OUT)
        GPIO.setup(self.in_a2, GPIO.OUT)
        GPIO.setup(self.en_b, GPIO.OUT)
        GPIO.setup(self.in_b1, GPIO.OUT)
        GPIO.setup(self.in_b2, GPIO.OUT)

        self.pwm_a = GPIO.PWM(self.en_a, 100)     # frequency Hz
        self.pwm_a.start(0)                       # duty cycle 0-100
        self.pwm_b = GPIO.PWM(self.en_b, 100)
        self.pwm_b.start(0)

    def move(self, *, speed=0.5, turn=0., time=0.):
        speed *= 100

        # Turn sensitivity depends on speed, from tests that were made the best values are:
        # speed: 25 - sens: 50, speed: 50 - sens: 80, speed: 75 - sens: 110, speed: 100 - sens: 140
        # From these values the formula that derives is: sens = 1.2 * speed + 20
        if speed == 0:
            turn *= 100
        elif speed > 0:
            turn *= 1.2 * speed + 20
        elif speed < 0:
            turn *= 1.2 * speed - 20

        right_speed = speed - turn
        left_speed = speed + turn

        # Maximum speed can be up to 100
        right_speed = 100 if right_speed > 100 else right_speed
        right_speed = -100 if right_speed < -100 else right_speed
        left_speed = 100 if left_speed > 100 else left_speed
        left_speed = -100 if left_speed < -100 else left_speed

        # Minimum speed that car robot can move
        min_speed = 25
        right_speed = min_speed if 0.01 < right_speed < min_speed else right_speed
        right_speed = -min_speed if -min_speed < right_speed < -0.01 else right_speed
        left_speed = min_speed if 0.01 < left_speed < min_speed else left_speed
        left_speed = -min_speed if -min_speed < left_speed < -0.01 else left_speed

        self.pwm_a.ChangeDutyCycle(abs(right_speed))
        self.pwm_b.ChangeDutyCycle(abs(left_speed))

        GPIO.output(self.in_a1, GPIO.HIGH) if right_speed > 0 else GPIO.output(self.in_a1, GPIO.LOW)
        GPIO.output(self.in_a2, GPIO.LOW) if right_speed > 0 else GPIO.output(self.in_a2, GPIO.HIGH)
        GPIO.output(self.in_b1, GPIO.HIGH) if left_speed > 0 else GPIO.output(self.in_b1, GPIO.LOW)
        GPIO.output(self.in_b2, GPIO.LOW) if left_speed > 0 else GPIO.output(self.in_b2, GPIO.HIGH)

        sleep(time)

    def stop(self, time=0):
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)
        sleep(time)


def main():
    motor.move(speed=0.8, turn=0.5, time=3)
    motor.stop()


if __name__ == '__main__':
    motor = Motor(21, 20, 16, 26, 13, 19)
    main()
