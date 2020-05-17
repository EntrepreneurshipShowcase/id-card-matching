import gpiozero as gz
import time
class Servo:
    def __init__(self, servo_pin, max_angle=20):
        self.max_angle = max_angle
        self.servo_pin = servo_pin
        self.servo = gz.Servo(servo_pin, max_pulse_width=(1450+10.6*self.max_angle)/(1e6), min_pulse_width=(1450-10.6*self.max_angle)/(1e6))
    def mid(self):
        self.servo = gz.Servo(self.servo_pin, max_pulse_width=(1450+10.6*self.max_angle)/(1e6), min_pulse_width=(1450-10.6*self.max_angle)/(1e6))
        self.servo.mid()
        time.sleep(0.3)
        del self.servo
    def high(self):
        self.servo = gz.Servo(self.servo_pin, max_pulse_width=(1450+10.6*self.max_angle)/(1e6), min_pulse_width=(1450-10.6*self.max_angle)/(1e6))
        self.servo.max()
        time.sleep(0.3)
        del self.servo
    def low(self):
        self.servo = gz.Servo(self.servo_pin, max_pulse_width=(1450+10.6*self.max_angle)/(1e6), min_pulse_width=(1450-10.6*self.max_angle)/(1e6))
        self.servo.min()
        time.sleep(0.3)
        del self.servo
