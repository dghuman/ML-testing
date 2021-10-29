import tensorflow as tf
import numpy as np
from DataSetGen.DataGen_2DRand import load_pickle

test_data = load_pickle(
    "/home/dilraj/Documents/ML-testing/DataSetGen/2D_data/data_2.pkl"
)
new_arr_t = []

for arr in test_data["arr"]:
    new_arr_t.append(arr.flatten())

test_labels = tf.convert_to_tensor(np.array((test_data["labels"])), dtype=tf.float32)
test_data = tf.convert_to_tensor(np.array((new_arr_t)), dtype=tf.float32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
test_dataset = test_dataset.batch(10)

model = tf.keras.models.load_model(
    "/home/dilraj/Documents/ML-testing/fit_models/model_test.tf"
)
