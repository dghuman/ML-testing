import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
from DataSetGen.DataGen_2DRand import load_pickle

outdir = "/home/dilraj/Documents/ML-testing/fit_models/plots"
history = load_pickle(
    "/home/dilraj/Documents/ML-testing/fit_models/model_test_history.pkl"
)


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle("Training Metrics")

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(history["loss"])

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(history["accuracy"])
plt.show()
