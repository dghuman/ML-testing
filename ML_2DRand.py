# Script where the tensorflow work will be done.
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from DataSetGen.DataGen_2DRand import load_pickle, save_pickle
from tools.timer import Timer  # load in Timer class

usage = "usage: %prog [options] inputfile"
parser = OptionParser(usage)
timer = Timer()
parser.add_option(
    "-din",
    "--datain",
    type="string",
    default="/home/dilraj/Documents/ML-testing/DataSetGen/2D_data/data_1.pkl",
    dest="DATA_IN",
    help="Training data input file in pickle format (.pkl).",
)
parser.add_option(
    "-tdin",
    "--testdata",
    type="string",
    default="/home/dilraj/Documents/ML-testing/DataSetGen/2D_data/data_2.pkl",
    dest="TEST_IN",
    help="Test data input file in pickle format (.pkl).",
)
parser.add_option(
    "-l",
    "--learn",
    type="bool",
    default=False,
    dest="LEARN",
    help="Boolian to determine if this is training that is continuing rather than starting anew.",
)
parser.add_option(
    "-mf",
    "--modelfile",
    type="string",
    default="/home/dilraj/Documents/ML-testing/fit_models/model_test.tf",
    dest="MODEL_OUT",
    help="Output model file saved in the tensorflow file type (.tf).",
)
parser.add_option(
    "-t",
    "--timer",
    type="bool",
    default=True,
    dest="TIME",
    help="Timer option. Set to True by default and times how long current training has been running.",
)
parser.add_option(
    "-h",
    "--history",
    type="string",
    default="/home/dilraj/Documents/ML-testing/fit_models/model_test_history.pkl",
    dest="HISTORY",
    help="History file containing information about the training stats saved in pickled format (.pkl).",
)
(options, args) = parser.parse_args()

""" 
First load in the data that will be used to train. Dictionary is loaded into data. Calls are 'arr' and 'labels'. The loaded data is then converted into tensors so that they can be zipped into a tensorflow data set that is parsable by the workflow.
"""
data = load_pickle(options.DATA_IN)
test_data = load_pickle(options.TEST_IN)

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
if options.LEARN:
    model = tf.keras.models.load_model(options.MODEL_OUT)
else:
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

if options.TIMER:
    timer.start()
History = model.fit(dataset, epochs=20000, verbose=1)
if options.TIMER:
    timer.stop()

history = {}
history = History.history

model.save(options.MODEL_OUT)
save_pickle(history, options.HISTORY)
