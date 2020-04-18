import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# define global parameters
# NUM_WORDS=8000
# SEQ_LEN=128
# EMBEDDING_SIZE=128
# BATCH_SIZE=128
# EPOCHS=5
# THRESHOLD=0.5
# TRAIN_DATA_URL = "https://raw.githubusercontent.com/colsoncrim/tensorflow-python/master/navigate_to_card_form%20-%20Sheet1.csv"
# TEST_DATA_URL = "https://raw.githubusercontent.com/colsoncrim/tensorflow-python/master/navigate_to_card_form_testing%20-%20Sheet1.csv"

# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

#load data
URL = 'https://raw.githubusercontent.com/colsoncrim/tensorflow-python/master/navigate_to_card_form%20-%20Sheet1.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

# split dataframe into training, testing, and validation data
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Create an input pipeline using tf.data
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('text') # the label for "Add new card" is "text". The label for "Delete card" is "other"
  # wrap dataframe with tf.data so we can use feature columns from the Pandas dataframe to features used to train the model 
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

  batch_size = 32 
  train_ds = df_to_dataset(train, batch_size=batch_size)
  val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
  test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of results:', feature_batch['result'])
  print('A batch of text:', label_batch )


  thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])

  thal_one_hot = feature_column.indicator_column(thal)
  demo(thal_one_hot)

  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# create and train the model
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='sigmoid'),
  layers.Dense(128, activation='sigmoid'),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
