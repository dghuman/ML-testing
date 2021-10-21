# Script where the tensorflow work will be done.
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from DataSetGen.DataGen_2DRand import load_pickle

''' 
First load in the data that will be used to train. Dictionary is loaded into data. Calls are 'arr' and 'labels'. The loaded data is then converted into tensors so that they can be zipped into a tensorflow data set that is parsable by the workflow.

'''
data = load_pickle("/home/dilraj/Documents/ML-testing/DataSetGen/2D_data/data_1.pkl")

train_labels = tf.convert_to_tensor(np.array((data["labels"])), dtype=tf.float32)
train_data = tf.convert_to_tensor(np.array((data["arr"])), dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))




# Will use a simple dense NN at first (dense meaning that the layers are completely connected, so every node is connected to every other node). This means a call to the simple layer.
