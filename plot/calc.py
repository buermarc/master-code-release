import pandas
import numpy as np
from matplotlib import pyplot as plt

out = pandas.read_csv("out.csv")

values = out[out.columns[(3 * 5) + 1 : (3 * 6) + 1]]

plt.plot(values)
plt.savefig("out.pdf")
