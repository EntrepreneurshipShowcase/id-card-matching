from aspen import aspen_pi

RED_LED_PIN = 0
GREEN_LED_PIN = 0
ADD_BUTTON_PIN = 0
VERIFY_BUTTON_PIN = 0
USE_CARD_VEC = False


def main():
    with aspen_pi.AspenPi(
        (ADD_BUTTON_PIN, VERIFY_BUTTON_PIN),
        (RED_LED_PIN, GREEN_LED_PIN),
        use_card_vec=USE_CARD_VEC,
    ) as aspen:
        aspen.run()


if __name__ == "__main__":
    main()