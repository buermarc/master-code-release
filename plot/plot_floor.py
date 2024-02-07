import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_json")

    args = parser.parse_args()

    with Path(args.experiment_json).open(mode="r", encoding="UTF-8") as _file:
        _json = json.loads(_file.read())
        normal_and_point = _json["floor"]
        normal = np.zeros((len(normal_and_point), 3))
        point = np.zeros((len(normal_and_point), 3))
        for idx, element in enumerate(normal_and_point):
            normal[idx, :] = element["normal"][0]
            point[idx, :] = element["point"][0]

        plt.plot(normal[:, 0], label="X")
        plt.plot(normal[:, 1], label="Y")
        plt.plot(normal[:, 2], label="Z")
        plt.title("floor normal")
        plt.legend()
        plt.show()
        plt.cla()

        plt.plot(point[:, 0], label="X")
        plt.plot(point[:, 1], label="Y")
        plt.plot(point[:, 2], label="Z")
        plt.title("floor point")
        plt.legend()
        plt.show()
        plt.cla()



if __name__ == "__main__":
    main()
