import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import cv2
import functools

import predict

class IDDatabase:
    def __init__(self, predictor):
        self.cred = credentials.Certificate("./id-matching-fbebc4da731a.json")
        firebase_admin.initialize_app(self.cred)
        self.db = firestore.client()
        self.predictor = predictor
        self.doc = self.db.collection(u"base").document(u"dhs")
    def add_person(self, name, id, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = {
            str(id): {
                u"name": name,
                u"id": id,
                u"vector": self.predictor.get_vec(image).tolist()[0],
            }
        }
        self.doc.update(data)

    def get_ids(self):
        return self.doc.get()


class DataDriver:
    def __init__(self):
        self.predictor = predict.Predictor()
        self.database = IDDatabase(self.predictor)
        # Immediately get all ids
        self.database.doc.on_snapshot(self.on_update)
        self.threshold = 95
    def on_update(self, doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            self.id_vectors = doc.to_dict()

    def lookup_person(self, comp_image):
        comp_image = cv2.cvtColor(comp_image, cv2.COLOR_BGR2RGB)
        comp_image_vec = self.predictor.get_vec(comp_image)[0]
        min_dis = float("inf")
        matched_id = None
        for i in self.id_vectors.keys():
            dis = self.predictor.get_distance(
                comp_image_vec, np.array(self.id_vectors[i]["vector"])
            )
            if dis < min_dis:
                matched_id = self.id_vectors[i]["id"], self.id_vectors[i]["name"]
                min_dis = dis
        return matched_id

    def verify(self, id, comp_image, debug=False):
        comp_image = cv2.cvtColor(comp_image, cv2.COLOR_BGR2RGB)
        comp_image_vec = self.predictor.get_vec(comp_image)[0]
        dis = self.predictor.get_distance(
                comp_image_vec, np.array(self.id_vectors[str(id)]["vector"])
        )
        if debug:
            print(dis)
        if dis < self.threshold:
            return True
        else:
            return False
    def add_person(self, name, id, image):
        self.database.add_person(name, id, image)