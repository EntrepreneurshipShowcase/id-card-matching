from io import BytesIO
from time import sleep
from picamera import PiCamera
from PIL import Image

import numpy as np

def take_picture(camera, size=(512, 512)):
    output = np.empty((512, 512, 3), dtype=np.uint8)
    camera.capture(output, 'bgr')
    return output

def get_camera():
    return PiCamera()