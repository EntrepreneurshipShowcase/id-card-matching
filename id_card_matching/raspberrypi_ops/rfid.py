import mfrc522
import numpy as np
import logging
import time
class RFID:
    def __init__(self, use_card_vec=False):
        self.reader = mfrc522.SimpleMFRC522()
        self.use_card_vec = use_card_vec
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass
    def write(self, id, name, vec=None):
        if self.use_card_vec and vec is not None:
            text = f"{id}/{name}/{np.array2string(vec)}"
        else:
            text = f"{id}/{name}"
        start=time.time()
        rfid, _ = self.reader.write(text)
        end=time.time()
        logging.debug(str(end-start))
        logging.info("Written")
        return rfid
    def read(self):
        rfid, text = self.reader.read()
        logging.info("Read RFID Successfully, text: " + text)
        # import pdb; pdb.set_trace()
        if self.use_card_vec:
            id, name, array = text.split("/")
            vec = np.fromstring(array)
        else:
            id, name = text.split("/")
            vec = None
        # array = "".join(array)
        return rfid, id, name, vec