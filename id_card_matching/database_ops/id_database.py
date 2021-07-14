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
        self.identification_doc = (
            self.db.collection(u"base")
            .document(u"dhs")
            .collection(u"identification")
            .document(u"vectors")
        )
        self.attendance_doc = (
            self.db.collection(u"base")
            .document(u"dhs")
            .collection(u"attendance")
            .document(u"overall")
        )
        self.attendance_collection = (
            self.db.collection(u"base")
            .document(u"dhs")
            .collection(u"attendance")
        )
    def add_person(self, name, id, vec, rfid=None):
        if rfid:
            id_num = rfid
        else:
            id_num = id
        # vec = self.predictor.get_vec(image)[0]
        data = {str(id_num): {u"name": name, u"id": id, u"vector": vec.tolist(),}}
        self.identification_doc.update(data)
        data = {str(id): {u"status": False}}
        self.attendance_doc.update(data)

    def update_attendance(self, id, status):
        data = {str(id): {u"status": status}}
        self.attendance_doc.update(data)

        # Adding query for different docs per student
        student_docs = self.attendance_collection.where(u"id", u"==", str(id)).limit(1).stream()
        for doc in student_docs:
            self.attendance_collection.document(doc.id).update({"status": status})

    def get_ids(self):
        return self.identification_doc.get()

    def get_all_attendance(self):
        return self.attendance_doc.get()

    def get_attendance(self, id):
        return self.get_all_attendance().to_dict().get(str(id), False)

    def update_all_attendance(self, status):
        attendance_data = self.get_all_attendance()
        for i in attendance_data.keys():
            attendance_data[i] = {
                list(attendance_data[i].keys())[0]: {u"status": status}
            }
        self.attendance_doc.update(attendance_data)


class DataDriver:
    def __init__(self, predictor=None):
        if predictor is None:
            self.predictor = predict.Predictor()
        else:
            self.predictor = predictor
        self.database = IDDatabase(self.predictor)
        # Immediately get all ids
        self.database.identification_doc.on_snapshot(self.on_id_update)
        self.database.attendance_doc.on_snapshot(self.on_attendance_update)
        
        self.threshold = self.predictor.threshold

    def on_id_update(self, doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            self.id_vectors = doc.to_dict()

    def on_attendance_update(self, doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            self.attendance = doc.to_dict()

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

    def verify_and_update(self, rfid, vec, debug=False):
        self.database.update_attendance(self.id_vectors[str(rfid)]["id"], True)
        return True
        return self.predictor.is_same_face(
            vec, np.array(self.id_vectors[str(rfid)]["vector"]), debug=debug
        )

    def add_person(self, name, id, vec, rfid=None):
        self.database.add_person(name, id, vec, rfid=rfid)

    def end_day(self):
        self.database.update_all_attendance(False)
    def get_attendance(self, id):
        return self.database.get_attendance(id)
    def update_attendance(self, id, status):
        self.database.update_attendance(id, status)
    def get_all_attendance(self):
        return self.database.get_all_attendance()