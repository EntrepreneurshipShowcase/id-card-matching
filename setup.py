from setuptools import setup
import setuptools
setup(name='aspen',
    version='1.0',
    install_requires=['tf-nightly', 'firebase-admin', 'opencv-python', 'pillow', 'tqdm', 'cmake', 'dlib', 'face-recognition', 'face-recognition-models'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    )