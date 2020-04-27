import dlib
import cv2

class FaceCrop:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
    def get_face_coor(self, image):
        faces = self.face_detector(image, 1)
        if len(faces) > 0:
            x = faces[0].left()
            y = faces[0].top()
            w = faces[0].right() - x
            h = faces[0].bottom() - y
            return (x, y, w, h)
        else:
            return None
    def crop(self, image):
        coor = self.get_face_coor(image)
        if coor:
            x, y, w, h = coor
            cropped_image = image[y:y+h, x:x+w]
        else:
            cropped_image = image
        return cropped_image