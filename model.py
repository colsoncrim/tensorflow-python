import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 
# import functools
# print(tf.__version__)


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







