from gpiozero import LED
import time

class LEDSet:
    def __init__(self, red_pin, green_pin):
        self.red_led = LED(red_pin)
        self.green_led = LED(green_pin)
    def success(self):
        self.red_led.on()
        time.sleep(2)
        self.red_led.off()
    def fail(self):
        self.green_led.on()
        time.sleep(2)
        self.green_led.off()