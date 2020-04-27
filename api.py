import facial_reid.id_database_fr as db

import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--image")
driver = db.DataDriver()



if __name__ == "__main__":
    args = parser.parse_args()
    image_file = args.image
    image = cv2.imread(image_file)
    id, name = driver.lookup_person(image)
    print(name)