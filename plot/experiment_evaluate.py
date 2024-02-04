import numpy as np
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
from enum import IntEnum
from scipy import signal


def old_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder")

    args = parser.parse_args()

    truth = np.load(f"{args.experiment_folder}/truth.npy")
    filtered = np.load(f"{args.experiment_folder}/filtered.npy")
    unfiltered = np.load(f"{args.experiment_folder}/unfiltered.npy")

    error_filtered = np.linalg.norm(truth - filtered, axis=2)
    rms_filtered = np.sqrt(np.mean(np.power(error_filtered, 2), axis=0))

    error_unfiltered = np.linalg.norm(truth - unfiltered, axis=2)
    rms_unfiltered = np.sqrt(np.mean(np.power(error_unfiltered, 2), axis=0))

    print(f"rms_filtered: {rms_filtered}")
    print(f"rms_unfiltered: {rms_unfiltered}")
    print(f"diff rms: {(rms_unfiltered - rms_filtered)}")

    com_ts =  np.load(f"{args.experiment_folder}/com_ts.npy")
    com =  np.load(f"{args.experiment_folder}/com.npy")
    com_unfiltered =  np.load(f"{args.experiment_folder}/com_unfiltered.npy")
    cop =  np.load(f"{args.experiment_folder}/cop.npy")

    # plt.plot(error_filtered,0], label="error filtered")
    # plt.plot(error_unfiltered, label="unfliltered filtered")

    plt.title("Wrist Z")
    plt.plot(truth[:, 2, 2], label="qtm wrist z")
    plt.plot(unfiltered[:, 2, 2], label="unfiltered wrist z")
    plt.plot(filtered[:, 2, 2], label="filtered wrist z")
    plt.legend()
    plt.show()
    plt.cla()


    plt.plot(com_ts, com[:,0], label="com filtered")
    plt.plot(com_ts, com_unfiltered[:,0], label="com unfiltered")
    plt.plot(com_ts, cop[:,0], label="cop qtm")
    plt.legend()
    plt.show()
    plt.cla()

    plt.plot(com_ts, com[:,1], label="com filtered")
    plt.plot(com_ts, com_unfiltered[:,1], label="com unfiltered")
    plt.plot(com_ts, cop[:,1], label="cop qtm")
    plt.legend()
    plt.show()
    plt.cla()

@dataclass
class Data:
    down_kinect_com: np.ndarray
    down_kinect_joints: np.ndarray
    down_kinect_ts: np.ndarray
    down_kinect_unfiltered_com: np.ndarray
    down_kinect_unfiltered_joints: np.ndarray
    down_qtm_cop: np.ndarray
    down_qtm_cop_ts: np.ndarray
    down_qtm_joints: np.ndarray
    down_qtm_ts: np.ndarray
    kinect_com: np.ndarray
    kinect_joints: np.ndarray
    kinect_ts: np.ndarray
    kinect_unfiltered_com: np.ndarray
    kinect_unfiltered_joints: np.ndarray
    qtm_cop: np.ndarray
    qtm_cop_ts: np.ndarray
    qtm_joints: np.ndarray
    qtm_ts: np.ndarray


Joint = IntEnum("Joint", [
    "PELVIS",
    "SPINE_NAVEL",
    "SPINE_CHEST",
    "NECK",
    "CLAVICLE_LEFT",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "HAND_LEFT",
    "HANDTIP_LEFT",
    "THUMB_LEFT",
    "CLAVICLE_RIGHT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HAND_RIGHT",
    "HANDTIP_RIGHT",
    "THUMB_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "HEAD",
    "NOSE",
    "EYE_LEFT",
    "EAR_LEFT",
    "EYE_RIGHT",
    "EAR_RIGHT",
], start = 0)


def load_processed_data(path: Path) -> Data:
    return Data(
        np.load(path / "down_kinect_com.npy"),
        np.load(path / "down_kinect_joints.npy"),
        np.load(path / "down_kinect_ts.npy"),
        np.load(path / "down_kinect_unfiltered_com.npy"),
        np.load(path / "down_kinect_unfiltered_joints.npy"),
        np.load(path / "down_qtm_cop.npy"),
        np.load(path / "down_qtm_cop_ts.npy"),
        np.load(path / "down_qtm_joints.npy"),
        np.load(path / "down_qtm_ts.npy"),
        np.load(path / "kinect_com.npy"),
        np.load(path / "kinect_joints.npy"),
        np.load(path / "kinect_ts.npy"),
        np.load(path / "kinect_unfiltered_com.npy"),
        np.load(path / "kinect_unfiltered_joints.npy"),
        np.load(path / "qtm_cop.npy"),
        np.load(path / "qtm_cop_ts.npy"),
        np.load(path / "qtm_joints.npy"),
        np.load(path / "qtm_ts.npy"),
    )

def double_filtered(data: np.ndarray, sample_frequency: int = 15, cut_off: int = 6, N: int = 2) -> np.ndarray:
    shape = data.shape
    if len(shape) == 1:
        return _double_filtered(data, sample_frequency, cut_off, N)
    elif len(shape) == 2:
        result = np.empty_like(data)
        for i in range(shape[1]):
            result[:, i] = _double_filtered(data[:, i], sample_frequency, cut_off, N)
        return result
    elif len(shape) == 3:
        # Bad performance, but hopefully not so important
        result = np.empty_like(data)
        for i in range(shape[1]):
            for j in range(shape[2]):
                result[:, i, j] = _double_filtered(data[:, i, j], sample_frequency, cut_off, N)
    else:
        print(f"shape: {shape}")
        raise NotImplementedError


def _double_filtered(data: np.ndarray, sample_frequency: int = 15, cut_off: int = 6, N: int = 2) -> np.ndarray:
    """Take Nx1 data and return it double filtered."""
    mean = data.mean()
    sos = signal.butter(N, cut_off, fs=sample_frequency, output="sos")
    once_filtered = signal.sosfilt(sos, data - mean)
    return np.flip(signal.sosfilt(sos, np.flip(once_filtered))) + mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder")

    args = parser.parse_args()

    data = load_processed_data(Path(args.experiment_folder))

    plt.plot(data.down_kinect_ts, data.down_kinect_com[:, 0], label="kinect_com")
    plt.plot(data.down_kinect_ts, double_filtered(data.down_kinect_com[:, 0]), label="butter kinect_com")
    plt.plot(data.qtm_cop_ts, data.qtm_cop[:, 0], label="qtm cop")
    plt.plot(data.qtm_cop_ts, double_filtered(data.qtm_cop[:, 0], 900), label="butter qtm cop")
    plt.legend()
    plt.show()
    plt.cla()

    plt.plot(data.kinect_ts, data.kinect_joints[:, int(Joint.SHOULDER_LEFT), 0], label="kinect")
    plt.plot(data.qtm_ts, data.qtm_joints[:, 0, 0], label="qtm")
    plt.legend()
    plt.show()
    plt.cla()

    plt.plot(data.kinect_ts, data.kinect_joints[:, int(Joint.ELBOW_LEFT), 0], label="kinect")
    plt.plot(data.qtm_ts, data.qtm_joints[:, 2, 0], label="qtm")
    plt.legend()
    plt.show()
    plt.cla()

    plt.plot(data.kinect_ts, data.kinect_joints[:, int(Joint.WRIST_LEFT), 0], label="kinect")
    plt.plot(data.qtm_ts, data.qtm_joints[:, 2, 0], label="qtm")
    plt.legend()
    plt.show()
    plt.cla()

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder")

    args = parser.parse_args()

    data = load_processed_data(Path(args.experiment_folder))

    out = double_filtered(data.qtm_cop, 900)
    assert np.all(out[:, 0] == double_filtered(data.qtm_cop[:, 0], 900))

    '''
    # Figure out how scipy correlate works
    # turns out 0..len(a) -> b starts before a
    # turns out len(a)+1..len(a)+len(b) -> a starts before b
    x = np.arange(128) / 128
    a = np.sin(2 * np.pi * x)
    b = np.sin(2 * np.pi * (x+0.05))
    corr = signal.correlate(a, b)
    plt.plot(a, label="a")
    plt.plot(b, label="b")
    plt.legend()
    plt.show()
    plt.cla()
    lags = signal.correlation_lags(len(a), len(b))
    corr /= np.max(corr)
    plt.plot(lags, corr)
    plt.show()
    breakpoint()
    '''

if __name__ == "__main__":
    # main()
    test()
