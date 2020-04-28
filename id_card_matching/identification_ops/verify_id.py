import numpy as np


def verify(predictor, camera, reader, card_vec=False, driver=None):
    rfid_id, id, name, vec = reader.read()
    image = camera.take_picture()
    comp_vec = predictor.get_vec(image)
    if card_vec and vec.size != 0:
        return predictor.is_same_person(vec, comp_vec)
    elif driver is not None:
        return driver.verify(rfid_id, comp_vec)
        