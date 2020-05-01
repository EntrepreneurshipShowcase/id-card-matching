import aspen.aspen_base as aspen
from raspberrypi_ops import button, led, camera
from random import random
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
class AspenPi(aspen.Aspen):
    def __init__(self, button_pins, led_pins, use_card_vec=False):
        super(AspenPi, self).__init__(use_card_vec=use_card_vec)
        verify_pin, add_pin = button_pins
        self.verify_button = button.Button(verify_pin)
        self.add_button = button.Button(add_pin)
        self.led = led.LEDSet(*led_pins)
    def run(self):
        try:
            self.verify_button.add_function(self.verify_id)
            self.add_button.add_function(self.add_random_id)
            while True:
                pass
        except KeyboardInterrupt:
            return
    def verify_id(self):
        is_person = super(AspenPi, self).verify()
        if is_person:
            self.led.success()
        else:
            self.led.fail()
    def add_random_id(self):
        pic = camera.take_picture(self.camera)
        rand_id = random.random()*1000 // 1
        super(AspenPi, self).add_person(None, rand_id, pic)
