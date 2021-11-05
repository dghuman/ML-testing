# 2021-Oct-10 Dilraj Ghuman
# This is a file to generate some simple 2D data to train on as practice

import random as rand
import numpy as np
import pickle


def save_pickle(obj, filename):
    with open(filename, "wb") as output:  # Overwrites
        pickle.dump(obj, output, -1)
    return 0


def load_pickle(filename):
    with open(filename, "rb") as input:
        return pickle.load(input)


def make_track(theta, intercept):
    # Angle to slope
    def slope():
        rise = np.sin(theta)
        run = np.cos(theta)
        return rise / run

    m = slope()
    # Feed track a numpy array, it'll spit out the y_values
    def track(x):
        y = m * x + intercept
        return y

    return track, m


# Need to compute intersections of grid with function. Suppose the grid is passed as numpy array of zeros. Size determines granularity in a 32 x 32 (pixel) setting. So, a 32 x 32 array has each entry represent a pixel. Return array filled with hit info.
def hit_array(empty_array, _track):
    dim = empty_array.shape[0]  # Read in dimension
    bin_array = np.linspace(0, 32, dim + 1)  # Get bin ranges that need to be checked
    for n in range(dim):
        xval = np.linspace(bin_array[n], bin_array[n + 1], 100)  # Make domain to check
        yval = _track(xval)  # Compute values over domain
        for m in range(dim):  # Loop over values y bins
            cond = np.logical_and(
                np.less_equal(yval, bin_array[m + 1]),
                np.greater_equal(yval, bin_array[m]),
            )  # Check if yval is in bin
            if cond.any():  # If it is in bin, fill the bin
                empty_array[m, n] = 1
    return 0


def main():
    # Data Structure to save the numpy array and direction
    data_set = {
        "labels": [],
        "arr": [],
    }
    n = 10000
    for i in range(n):  # Make n entries
        hits = np.zeros([32, 32])
        b = rand.uniform(0, 32)
        angle = rand.uniform(0, 2 * np.pi)
        track, m = make_track(angle, b)
        hit_array(hits, track)
        if np.count_nonzero(hits) < 3: # If less than 3 hits, skip it
            continue 
        data_set["labels"].append(
            np.array(([b, m]))
        )  # Appended as intercept then slope
        data_set["arr"].append(hits)
    save_pickle(
        data_set, "/home/dilraj/Documents/ML-testing/DataSetGen/2D_data/data_1.pkl"
    )


if __name__ == "__main__":
    main()
