import numpy as np

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import face_recognition
import cv2
class IDDatabase:
    def __init__(self):
        self.cred = credentials.Certificate("./id-matching-fbebc4da731a.json")
        firebase_admin.initialize_app(self.cred)
        self.db = firestore.client()
        self.doc = self.db.collection(u"base").document(u"dhs")
    def add_person(self, name, id, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = {
            str(id): {
                u"name": name,
                u"id": id,
                u"vector": face_recognition.face_encodings(image)[0].tolist(),
            }
        }
        self.doc.update(data)

    def get_ids(self): 
        return self.doc.get()



class DataDriver:
    def __init__(self):
        self.database = IDDatabase()
        self.threshold = 95
        self.database.doc.on_snapshot(self.on_update)

    def on_update(self, doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            self.id_vectors = doc.to_dict()

    def lookup_person(self, comp_image):
        comp_image = cv2.cvtColor(comp_image, cv2.COLOR_BGR2RGB)
        comp_image_vec = face_recognition.face_encodings(comp_image)
        if len(comp_image_vec) > 0:
            comp_image_vec = comp_image_vec[0]
        else:
            return False, False
        min_dis = float("inf")
        matched_id = None
        # import ipdb; ipdb.set_trace()
        for i in self.id_vectors.keys():
            dis = face_recognition.face_distance(np.array([self.id_vectors[i]["vector"]]), comp_image_vec)
            if dis < min_dis:
                matched_id = self.id_vectors[i]["id"], self.id_vectors[i]["name"]
                min_dis = dis
        return matched_id
    def verify_id(self, id, comp_image, debug=False):
        comp_image = cv2.cvtColor(comp_image, cv2.COLOR_BGR2RGB)
        comp_image_vec = face_recognition.face_encodings(comp_image)[0]
        id_vec = self.id_vectors[str(id)]["vector"]
        dis = face_recognition.face_distance(
                np.array([id_vec]), comp_image_vec
        )
        if debug:
            print(dis)
        if dis < self.threshold:
            return True
        else:
            return False
    def add_person(self, name, id, image):
        self.database.add_person(name, id, image)