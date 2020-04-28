from io import BytesIO
from time import sleep
from picamera import PiCamera
from PIL import Image

import numpy as np

def take_picture(camera, size=(512, 512)):
    camera.resolution = size
    camera.framerate = 24
    sleep(2)
    output = np.empty((240, 320, 3), dtype=np.uint8)
    camera.capture(output, 'brg')
    return output

def get_camera():
    return PiCamera()