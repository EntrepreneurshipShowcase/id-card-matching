import gpiozero as gz

class Servo:
    def __init__(self, servo_pin, max_angle=45):
        self.max_angle = max_angle
        self.servo = gz.Servo(servo_pin, max_pulse_width=(1450+10.6*max_angle)/(1e6), min_pulse_width=(1450-10.6*max_angle)/(1e6))
    def mid(self):
        self.servo.mid()
    def high(self):
        self.servo.max()
    def low(self):
        self.servo.min()