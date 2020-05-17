from database_ops import id_database
from dl import predict_fr as predict
from identification_ops import add_person, verify_id
import logging

from raspberrypi_ops import camera, rfid, servo
import logging
from time import sleep
import dlib
class Aspen:
    def __init__(self, use_card_vec=False):
        self.predictor = predict.Predictor()
        if use_card_vec:
            self.driver = None
        else:
            logging.info("Initializing Database")
            self.driver = id_database.DataDriver()
        logging.info("Initializing camera")
        self.camera = camera.get_camera()
        self.camera.resolution = (512, 512)
        self.camera.framerate = 24
        sleep(2)
        logging.info("Initializing RFID reader")
        self.use_card_vec = use_card_vec
        self.reader = rfid.RFID(use_card_vec=use_card_vec)
        self.detector = dlib.get_frontal_face_detector()
    def add_person(self, name, id, image):
        logging.info("Waiting for rfid chip to write")
        add_person.add_person(self.predictor, self.reader, name, id, image, driver=self.driver)
    def verify(self, servo):
        logging.info("waiting for rfid chip read")
        return True
        # return verify_id.verify(self.predictor, self.camera, self.reader, self.detector, servo, card_vec=self.use_card_vec, driver=self.driver)
    def __enter__(self):
        self.reader.__enter__()
    def __exit__(self, type, value, tb):
        self.reader.__exit__(type, value, tb)
        logging.info("Done, exiting...")
