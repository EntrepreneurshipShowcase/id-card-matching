import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm

NUM_EPOCHS = 400
BATCH_SIZE = 16
TRAIN_STEPS = 500
VAL_STEPS = 100
LEARNING_RATE = 0.0001
MIXED_PRECISION = False

DATA_DIR = "./data/lfwcrop_color/faces/"  # "./Data/"
NUM_FOLDERS = 1680  # 10575

DISTRIBUTE = False

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_triplet_data():
    while True:
        face_id = np.random.randint(0, NUM_FOLDERS)
        negative_id = np.random.randint(0, NUM_FOLDERS)
        anchor = np.random.choice(os.listdir("{}{:05d}".format(DATA_DIR, face_id)))
        positive = np.random.choice(
            os.listdir("{}{:05d}".format(DATA_DIR, face_id))
        )
        negative = np.random.choice(
            os.listdir("{}{:05d}".format(DATA_DIR, negative_id))
        )
        negative_path = "{:05d}".format(negative_id)
        face_path = "{:05d}".format(face_id)

        negative_image = tf.io.read_file(f"{DATA_DIR}{negative_path}/{negative}")
        anchor_image = tf.io.read_file(f"{DATA_DIR}{face_path}/{anchor}")
        positive_image = tf.io.read_file(f"{DATA_DIR}{face_path}/{positive}")

        yield (anchor_image, positive_image, negative_image)



def image_example(anchor, positive, negative):
    feature = {
        'anchor': _bytes_feature(anchor),
        'positive': _bytes_feature(positive),
        "negative": _bytes_feature(negative),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))
def main():
  record_file = 'triplet_data_test.tfrecords'

  data_iterator = get_triplet_data()

  num_save = 100000
  idx = 0
  with tf.io.TFRecordWriter(record_file) as writer:
      for anchor, positive, negative in tqdm(data_iterator, total=num_save):
          idx += 1
          tf_example = image_example(anchor, positive, negative)
          writer.write(tf_example.SerializeToString())
          if idx > num_save:
              break
if __name__ == "__main__":
  main()