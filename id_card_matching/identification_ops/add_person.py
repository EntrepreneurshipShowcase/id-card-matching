import numpy as np
import logging
def add_person(predictor, reader, name, id, image, driver=None):
    rfid_id = reader.write(id, name)
    vec = predictor.get_vec(image)
    if driver is not None:
        driver.add_person(name, id, vec, rfid=rfid_id)
