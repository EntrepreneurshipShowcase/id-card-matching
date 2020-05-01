import mfrc522
import numpy as np

class RFID:
    def __init__(self):
        pass
    def __enter__(self):
        self.reader = mfrc522.SimpleMFRC522()
    def __exit__(self, type, value, traceback):
        pass
    def write(self, id, name, vec=None):
        if vec is not None:
            text = f"{id}/{name}/{np.array2string(vec)}"
        else:
            text = f"{id}/{name}/{np.array2string(np.empty(0))}"
        id, _ = self.reader.write(text)
        return id
    def read(self):
        rfid, text = self.reader.read()
        id, name, * array = text.split("/")
        array = "".join(array)
        vec = np.fromstring(array)
        return rfid, id, name, vec