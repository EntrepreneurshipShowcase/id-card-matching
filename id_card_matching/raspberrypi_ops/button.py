import gpiozero as gz

class Button:
    def __init__(self, button_pin):
        self.button = gz.Button(button_pin)
    def add_function(self, func):
        self.button.when_pressed = func