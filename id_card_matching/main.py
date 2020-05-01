from aspen import aspen_pi

RED_LED_PIN =26 
GREEN_LED_PIN =16 
ADD_BUTTON_PIN =17 
VERIFY_BUTTON_PIN =27 
USE_CARD_VEC = False


def main():
    with aspen_pi.AspenPi(
        (ADD_BUTTON_PIN, VERIFY_BUTTON_PIN),
        (RED_LED_PIN, GREEN_LED_PIN),
        use_card_vec=USE_CARD_VEC,
    ) as aspen:
        print("initialized, begin playing") 
        aspen.run()


if __name__ == "__main__":
    main()
