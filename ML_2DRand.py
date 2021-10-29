# Script where the tensorflow work will be done.
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from DataSetGen.DataGen_2DRand import load_pickle, save_pickle
from tools.timer import Timer  # load in Timer class

timer = Timer()

""" 
First load in the data that will be used to train. Dictionary is loaded into data. Calls are 'arr' and 'labels'. The loaded data is then converted into tensors so that they can be zipped into a tensorflow data set that is parsable by the workflow.
"""
data = load_pickle("/home/dilraj/Documents/ML-testing/DataSetGen/2D_data/data_1.pkl")
test_data = load_pickle(
    "/home/dilraj/Documents/ML-testing/DataSetGen/2D_data/data_2.pkl"
)

new_arr = []
new_arr_t = []
for arr in data["arr"]:
    new_arr.append(arr.flatten())

for arr in test_data["arr"]:
    new_arr_t.append(arr.flatten())

train_labels = tf.convert_to_tensor(np.array((data["labels"])), dtype=tf.float32)
train_data = tf.convert_to_tensor(np.array((new_arr)), dtype=tf.float32)

test_labels = tf.convert_to_tensor(np.array((test_data["labels"])), dtype=tf.float32)
test_data = tf.convert_to_tensor(np.array((new_arr_t)), dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
dataset = dataset.batch(10)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
test_dataset = test_dataset.batch(10)

"""
Will use a simple dense NN at first (dense meaning that the layers are completely connected, so every node is connected to every other node). Sequential model does this well:
"""

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(1024,)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(2),
    ]
)

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.RMSprop()

model.compile(optimizer=optimizer, loss=loss_object, metrics=["accuracy"])

timer.start()
History = model.fit(dataset, epochs=10000, verbose=1)
timer.stop()

history = {}
history = History.history

model.save("/home/dilraj/Documents/ML-testing/fit_models/model_test.tf")
save_pickle(
    history, "/home/dilraj/Documents/ML-testing/fit_models/model_test_history.pkl"
)
