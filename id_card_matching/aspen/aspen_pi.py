import aspen.aspen_base as aspen
from raspberrypi_ops import button, led, camera, servo
from random import random
import logging
from signal import pause
class AspenPi(aspen.Aspen):
    def __init__(self, button_pins, led_pins, servo_pin, use_card_vec=False):
        super(AspenPi, self).__init__(use_card_vec=use_card_vec)
        verify_pin, add_pin = button_pins
        self.verify_button = button.Button(verify_pin)
        self.add_button = button.Button(add_pin)
        self.led = led.LEDSet(*led_pins)
        self.servo = servo.Servo(servo_pin)
    def run(self):
        try:
            logging.info("Ready")
            self.verify_button.add_function(self.verify_id)
            self.add_button.add_function(self.add_random_id)
            self.verify_id()
            pause()
        except KeyboardInterrupt:
            return
    def verify_id(self):
        is_person = super(AspenPi, self).verify(self.servo)
        if is_person:
            self.led.success()
            logging.info("verified")
        else:
            self.led.fail()
            logging.info("Failed")
    def add_random_id(self):
        pic = camera.take_picture(self.camera)
        logging.info("Took pic")
        inp_id = input("Enter id of the person")
        inp_name = input("Enter name of the person")
        super(AspenPi, self).add_person(inp_name, inp_id, pic)
        logging.info("Added")
