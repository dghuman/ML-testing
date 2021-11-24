import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
from DataSetGen.DataGen_2DRand import load_pickle
from optparse import OptionParser

usage = "usage: %prog [options] inputfile"
parser = OptionParser(usage)
parser.add_option(
    "-d",
    "--dirout",
    type="string",
    default="/home/dilraj/Documents/ML-testing/fit_models/plots/",
    dest="DIROUT",
    help="Output Directory.",
)
parser.add_option(
    "-i",
    "--input",
    type="string",
    default="/home/dilraj/Documents/ML-testing/fit_models/model_test_history.pkl",
    dest="INPUT",
    help="History input file in pickle format (.pkl).",
)
parser.add_option(
    "-o",
    "--output",
    type="string",
    default="out.png",
    dest="OUTPUT",
    help="Output file where image of plot is saved.",
)
parser.add_option(
    "-s",
    "--save",
    action="store_true",
    default=False,
    dest="SAVE",
    help="Whether or not the output plot should be saved. If not saved, it is only displayed.",
)
(options, args) = parser.parse_args()

outdir = options.DIROUT
history = load_pickle(options.INPUT)

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle("Training Metrics")


axes[0].set_ylabel("Loss", fontsize=14)
axes[0].grid(True)
axes[0].plot(history["loss"])

axes[1].set_ylabel("Mean Absolute Error", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].grid(True)
axes[1].plot(history["mean_absolute_error"])

if options.SAVE:
    plt.savefig(outdir + options.OUTPUT)
else:    
    plt.show()
