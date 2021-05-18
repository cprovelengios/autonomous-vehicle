#!/usr/bin/python3.7
import time
import RPi.GPIO as GPIO


# Class for measure distance with sensor SRF05
# The document I am referring throught the code is https://www.robot-electronics.co.uk/htm/srf05tech.htm
class SRF05:
    def __init__(self, trigger_pin, echo_pin):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.trigger_time = 0

        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)

    def measure(self):
        now = self.time_us()

        # The SRF05 can be triggered as fast as every 50ms, or 20 times each second.
        # You should wait 50ms before the next trigger, even if the SRF05 detects a close object and the echo pulse is shorter.
        # This is to ensure the ultrasonic "beep" has faded away and will not cause a false echo on the next ranging.
        pause = 50000 - (now - self.trigger_time)

        if pause > 0:
            self.sleep_us(pause)

        self.trigger()
        self.trigger_time = self.time_us()

        # The SRF05 will send out an 8 cycle burst of ultrasound at 40khz and raise its echo line high.
        # Wait no longer than 30ms.
        if GPIO.wait_for_edge(self.echo_pin, GPIO.RISING, timeout=30) is None:
            return None

        start = self.time_us()

        # Measure pulse duration, again do not wait more than 30ms.
        # If nothing is detected then the SRF05 will lower its echo line anyway after about 30ms.
        if GPIO.wait_for_edge(self.echo_pin, GPIO.FALLING, timeout=30) is None:
            return None

        end = self.time_us()
        width = end - start

        # With that logic we should not have real measurement with pulse longer than 30ms anyway
        if width > 30000:
            return None

        # If the width of the pulse is measured in us, then dividing by 58 will give you the distance in cm,
        # or dividing by 148 will give the distance in inches. us/58=cm or us/148=inches.
        return int(width / 58)

    def trigger(self):
        # Only need to supply a short 10us pulse to the trigger input to start the ranging.
        GPIO.output(self.trigger_pin, 1)
        self.sleep_us(10)
        GPIO.output(self.trigger_pin, 0)

    # Return time in microseconds
    def time_us(self):
        return int(time.time() * 1000000)

    def sleep_us(self, us):
        time.sleep(us / 1000000.0)


if __name__ == '__main__':
    sensor = SRF05(trigger_pin=23, echo_pin=24)

    try:
        while True:
            print(f'Measured Distance = {sensor.measure()} cm')
    except KeyboardInterrupt:
        print("Measurement stopped by User")
