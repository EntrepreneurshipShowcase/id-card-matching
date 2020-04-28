import numpy as np

def add_person(predictor, reader, name, id, image, driver=None):
    vec = predictor.get_vec(image)
    rfid_id = reader.write(id, name, vec=vec)
    if driver is not None:
        driver.add_person(name, id, vec, rfid=rfid_id)
