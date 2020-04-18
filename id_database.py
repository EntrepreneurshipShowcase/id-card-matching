import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import predict

import functools


class IDDatabase:
    def __init__(self, predictor):
        self.cred = credentials.Certificate("./id-matching-fbebc4da731a.json")
        firebase_admin.initialize_app(self.cred)
        self.db = firestore.client()
        self.predictor = predictor

    def add_person(self, name, id, image):
        data = {
            str(id): {
                u"name": name,
                u"id": id,
                u"vector": self.predictor.get_vec(image).tolist()[0],
            }
        }
        self.db.collection(u"base").document(u"dhs").update(data)

    def get_ids(self):
        self.doc = self.db.collection(u"base").document(u"dhs").get()
        return self.doc


class DataDriver:
    def __init__(self):
        self.predictor = predict.Predictor()
        self.database = IDDatabase(self.predictor)
        # Immediately get all ids
        self.id_vectors = self.database.get_ids().to_dict()

    def on_update(self, doc_snapshot, changes, read_time):
        pass

    def lookup_person(self, comp_image):
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

    def add_person(self, name, id, image):
        self.database.add_person(name, id, image)

