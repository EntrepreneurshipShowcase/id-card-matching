import database_ops.id_database as db

import cv2
import numpy as np
import argparse

import dl.predict

parser = argparse.ArgumentParser()
parser.add_argument("--image")
driver = db.DataDriver()
predictor = dl.predict.Predictor()

def main():
    args = parser.parse_args()
    image_file = args.image
    image = cv2.imread(image_file)
    vec = predictor.get_vec(image)
    if image is None:
        print("Error, exiting")
    id, name = driver.lookup_person(vec)
    print(name)
if __name__ == "__main__":
    main()