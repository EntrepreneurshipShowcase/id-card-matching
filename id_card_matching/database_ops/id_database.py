import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import cv2
import functools

import dl.predict_fr as predict
# import dl.predict

class IDDatabase:
    def __init__(self, predictor):
        self.cred = credentials.Certificate("./id-matching-fbebc4da731a.json")
        firebase_admin.initialize_app(self.cred)
        self.db = firestore.client()
        self.predictor = predictor
        self.doc = self.db.collection(u"base").document(u"dhs")
    def add_person(self, name, id, vec, rfid=None):
        if rfid:
            id_num = rfid
        else:
            id_num = id
        # vec = self.predictor.get_vec(image)[0]
        data = {
            str(id_num): {
                u"name": name,
                u"id": id,
                u"vector": vec.tolist(),
            }
        }
        self.doc.update(data)
    def get_ids(self):
        return self.doc.get()


class DataDriver:
    def __init__(self, predictor=None):
        if predictor is None:
            self.predictor = predict.Predictor()
        else:
            self.predictor = predictor
        self.database = IDDatabase(self.predictor)
        # Immediately get all ids
        self.database.doc.on_snapshot(self.on_update)
        self.threshold = self.predictor.threshold
    def on_update(self, doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            self.id_vectors = doc.to_dict()

    def lookup_person(self, vec):
        min_dis = float("inf")
        matched_id = None
        for i in self.id_vectors.keys():
            dis = self.predictor.get_distance(
                vec, np.array(self.id_vectors[i]["vector"])
            )
            if dis < min_dis:
                matched_id = self.id_vectors[i]["id"], self.id_vectors[i]["name"]
                min_dis = dis
        return matched_id

    def verify(self, id, vec, debug=False):
        return self.predictor.is_same_face(
                vec, np.array(self.id_vectors[str(id)]["vector"]), debug=debug
        )
    def add_person(self, name, id, vec, rfid=None):
        self.database.add_person(name, id, vec, rfid=rfid)
