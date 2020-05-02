from aspen import aspen_pi
import RPi.GPIO as GPIO
import logging
RED_LED_PIN = 26 
GREEN_LED_PIN = 16 
ADD_BUTTON_PIN = 17 
SERVO_PIN = 12
VERIFY_BUTTON_PIN = 27 
USE_CARD_VEC = False
GPIO.setmode(GPIO.BCM)

logging.basicConfig(level=logging.INFO)
def main():
    try:
        base = aspen_pi.AspenPi(
            (VERIFY_BUTTON_PIN, ADD_BUTTON_PIN),
            (RED_LED_PIN, GREEN_LED_PIN),
            SERVO_PIN,
            use_card_vec=USE_CARD_VEC,
        )
        logging.info("initialized, begin playing")
        # import ipdb; ipdb.set_trace()
        base.run()
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
