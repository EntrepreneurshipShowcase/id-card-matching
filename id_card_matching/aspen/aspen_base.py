from database_ops import id_database
from dl import predict_fr as predict
from identification_ops import add_person, verify_id
from raspberrypi_ops import camera, rfid

class Aspen:
    def __init__(self, use_card_vec=False):
        self.predictor = predict.Predictor()
        if use_card_vec:
            self.driver = id_database.DataDriver()
        else:
            self.driver = None
        self.camera = camera.get_camera()
        self.use_card_vec = use_card_vec
        self.reader = rfid.RFID()
    def add_person(self, name, id, image):
        print("Waiting for rfid chip") 
        add_person.add_person(self.predictor, self.reader, name, id, image, driver=self.driver)
    def verify(self):
        print("waiting for rfid chip read") 
        return verify_id.verify(self.predictor, self.camera, self.reader, card_vec=self.use_card_vec, driver=self.driver)
    def __enter__(self):
        self.reader.__enter__()
    def __exit__(self, type, value, tb):
        self.reader.__exit__(type, value, tb)
        print("Done, exiting...")
