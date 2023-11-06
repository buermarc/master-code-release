import numpy as np
from matplotlib import pyplot as plt
import pandas
from pathlib import Path

data_dir = Path("../data")

truth = pandas.read_csv(data_dir / "true_result.csv")
noisy = pandas.read_csv(data_dir / "noisy_result.csv")
filtered = pandas.read_csv(data_dir / "noisy_result.csv")
