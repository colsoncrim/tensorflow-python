import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 
import functools

# print(tf.__version__)

TRAIN_DATA_URL = "https://raw.githubusercontent.com/colsoncrim/tensorflow-python/master/navigate_to_card_form%20-%20Sheet1.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)



# encoder = info.features['text'].encoder

# print('Vocabulary size: {}'.format(encoder.vocab_size))







