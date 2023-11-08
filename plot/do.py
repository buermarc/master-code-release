from matplotlib import pyplot as plt
import numpy as np

import argparse

def main(name: str, out_name: str):
    with open(name, mode="r", encoding="utf-8") as _file:
        values = []
        for line in _file.readlines():
            values.append(float(line))

    plt.plot(np.arange(len(values)), values)
    plt.savefig(out_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("out_name")
    args = parser.parse_args()
    main(args.name, args.out_name) 
