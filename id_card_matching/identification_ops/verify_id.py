import numpy as np
from raspberrypi_ops import camera as cam
import logging
import time
def verify(predictor, camera, reader, detector, servo, card_vec=False, driver=None):
    rfid_id, id, name, vec = reader.read()
    logging.info("Taking picture normal...")
    image = cam.take_picture(camera)

    if len(detector(image)) == 0:
        servo.high()
        time.sleep(2)
        logging.info("Taking high photo")
        image = cam.take_picture(camera)
        if len(detector(image)) == 0:
            servo.low()
            time.sleep(2)
            logging.info("Taking low photo")
            image = cam.take_picture(camera)
    servo.mid()
    logging.info("Took pic")
    comp_vec = predictor.get_vec(image)
    if card_vec and vec is not None:
        return predictor.is_same_person(vec, comp_vec)
    elif driver is not None:
        return driver.verify_and_update(rfid_id, comp_vec)