import numpy as np
from matplotlib import pyplot as plt

import pandas


matlab = pandas.read_csv("../../matlab/file1.csv")
cpp = pandas.read_csv("../data/out.csv")
unfiltered = pandas.read_csv("../data/unfiltered_out.csv")

for joint_idx in range(32):
    fig, axis = plt.subplots(3, 2)
    mat_values = matlab[matlab.columns[(3 * joint_idx) : (3 * (joint_idx+1))]]

    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    axis[0, 0].plot(mat_values)
    axis[0, 0].set_title(f"Matlab {joint_idx}")

    values = cpp[cpp.columns[(3 * joint_idx) : (3 * (joint_idx+1))]]

    result = mat_values - values

    axis[1, 0].plot(values)
    axis[1, 0].set_title(f"Cpp {joint_idx}")

    un_values = unfiltered[unfiltered.columns[(3 * joint_idx) : (3 * (joint_idx+1))]]
    axis[2, 0].plot(un_values)
    axis[2, 0].set_title(f"Unfiltered {joint_idx}")

    axis[0, 1].plot(result[result.columns[:]], marker=",", alpha=0.5)
    axis[0, 1].set_title(f"Diff Cpp - Mat {joint_idx}")

    axis[1, 1].plot(values - un_values, marker=",", alpha=0.5)
    axis[1, 1].set_title(f"Diff Cpp - Unfiltered {joint_idx}")

    axis[2, 1].plot(mat_values - un_values, marker=",", alpha=0.5)
    axis[2, 1].set_title(f"Diff Matlab - Unfiltered {joint_idx}")

    plt.savefig(f"results/out-{joint_idx}.pdf")
    plt.close()
