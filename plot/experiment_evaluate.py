from __future__ import annotations
from pprint import pprint as pp
import os
import json
import numpy as np
from numpy.testing import assert_allclose
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
from enum import IntEnum
from scipy import signal
import numba


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
    down_kinect_velocities: np.ndarray
    down_qtm_cop: np.ndarray
    down_qtm_cop_ts: np.ndarray
    down_qtm_joints: np.ndarray
    down_qtm_ts: np.ndarray
    kinect_com: np.ndarray
    kinect_joints: np.ndarray
    kinect_ts: np.ndarray
    kinect_unfiltered_com: np.ndarray
    kinect_unfiltered_joints: np.ndarray
    kinect_velocities: np.ndarray
    qtm_cop: np.ndarray
    qtm_cop_ts: np.ndarray
    qtm_joints: np.ndarray
    qtm_ts: np.ndarray
    config: dict[str, str]


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

@numba.jit(nopython=True)
def _downsample(data: np.ndarray, timestamps: np.ndarray, target_frequency: int, downsampled_values: np.ndarray) -> np.ndarray:
    frame_duration = 1. / target_frequency

    down_i = 0
    for i in range(0, len(data)):
        next_frame_ts = frame_duration * down_i
        if timestamps[i] == next_frame_ts:
            downsampled_values[down_i] = data[i]
            down_i += 1
        if timestamps[i] < next_frame_ts:
            continue;
        if timestamps[i] > next_frame_ts:
            before_ts = timestamps[i-1]
            current_ts = timestamps[i]
            before_value = data[i-1]
            current_value = data[i]
            value = before_value + ((current_value - before_value) * ((next_frame_ts - before_ts) / (current_ts - before_ts)))
            downsampled_values[down_i] = value
            down_i += 1

def downsample(data: np.ndarray, timestamps: np.ndarray, target_frequency: int) -> np.ndarray:

    assert len(data) == len(timestamps)

    frame_duration = 1. / target_frequency
    downsampled_values_length = int(timestamps[-1] / frame_duration) + 1

    # Assume the first axis is the frame axis
    shape = list(data.shape)
    shape[0] = downsampled_values_length
    downsampled_values = np.zeros(shape)

    _downsample(data, timestamps, target_frequency, downsampled_values)

    return downsampled_values


def load_processed_data(path: Path) -> Data:
    return Data(
        np.load(path / "down_kinect_com.npy"),
        np.load(path / "down_kinect_joints.npy"),
        np.load(path / "down_kinect_ts.npy"),
        np.load(path / "down_kinect_unfiltered_com.npy"),
        np.load(path / "down_kinect_unfiltered_joints.npy"),
        np.load(path / "down_kinect_velocities.npy"),
        np.load(path / "down_qtm_cop.npy"),
        np.load(path / "down_qtm_cop_ts.npy"),
        np.load(path / "down_qtm_joints.npy"),
        np.load(path / "down_qtm_ts.npy"),
        np.load(path / "kinect_com.npy"),
        np.load(path / "kinect_joints.npy"),
        np.load(path / "kinect_ts.npy"),
        np.load(path / "kinect_unfiltered_com.npy"),
        np.load(path / "kinect_unfiltered_joints.npy"),
        np.load(path / "kinect_velocities.npy"),
        np.load(path / "qtm_cop.npy"),
        np.load(path / "qtm_cop_ts.npy"),
        np.load(path / "qtm_joints.npy"),
        np.load(path / "qtm_ts.npy"),
        json.load((path / "config.json").open(mode="r", encoding="UTF-8")),
    )

def double_butter(data: np.ndarray, sample_frequency: int = 15, cutoff: int = 6, N: int = 2, once: bool = False) -> np.ndarray:
    shape = data.shape
    if len(shape) == 1:
        return _double_butter(data, sample_frequency, cutoff, N, once)
    elif len(shape) == 2:
        result = np.empty_like(data)
        for i in range(shape[1]):
            result[:, i] = _double_butter(data[:, i], sample_frequency, cutoff, N, once)
        return result
    elif len(shape) == 3:
        # Bad performance, but hopefully not so important
        result = np.empty_like(data)
        for i in range(shape[1]):
            for j in range(shape[2]):
                result[:, i, j] = _double_butter(data[:, i, j], sample_frequency, cutoff, N, once)
        return result
    else:
        print(f"shape: {shape}")
        raise NotImplementedError


def _double_butter(data: np.ndarray, sample_frequency: int = 15, cutoff: int = 1, N: int = 2, once: bool = False) -> np.ndarray:
    """Take Nx1 data and return it double filtered."""
    mean = data[0]
    sos = signal.butter(N, cutoff, fs=sample_frequency, output="sos")
    once_filtered = signal.sosfilt(sos, data - mean)
    if once:
        return once_filtered + mean
    flip = np.flip(once_filtered)
    second_mean = flip[0]
    return np.flip(signal.sosfilt(sos, flip - second_mean) + second_mean) + mean


def compare_qtm_joints_kinect_joints(data: Data, cutoff: float = 0.15) -> tuple[float, float, float, float]:
    kinect_joints = [int(element) for element in [Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT]]

    corr = 0
    corr_un = 0

    rmse = 0
    rmse_un = 0
    for kinect_joint, qtm_joint in zip(kinect_joints, [0, 1, 2]):
        down_qtm = downsample(double_butter(data.qtm_joints[:, qtm_joint, :], 900), data.qtm_ts, 15)
        length = min(down_qtm.shape[0], data.down_kinect_joints.shape[0])
        o = max(int(length * cutoff), 1)

        down_qtm = down_qtm[:length][o:-o]
        joints = data.down_kinect_joints[:, kinect_joint,:][:length][o:-o]
        joints_un = data.down_kinect_unfiltered_joints[:, kinect_joint, :][:length][o:-o]

        corr += np.correlate(down_qtm[:, 0], joints[:, 0])[0] + np.correlate(down_qtm[:, 1], joints[:, 1])[0] + np.correlate(down_qtm[:, 2], joints[:, 2])[0]
        corr_un += np.correlate(down_qtm[:, 0], joints_un[:, 0])[0] + np.correlate(down_qtm[:, 1], joints_un[:, 1])[0] + np.correlate(down_qtm[:, 2], joints[:, 2])[0]

        diff = np.linalg.norm(down_qtm - joints, axis=1)
        rmse += np.sqrt(np.mean(np.power(diff, 2)))

        diff_un = np.linalg.norm(down_qtm - joints_un, axis=1)
        rmse_un += np.sqrt(np.mean(np.power(diff_un, 2)))

    return corr, corr_un, rmse, rmse_un


def compare_qtm_cop_kinect_cop(data: Data, cutoff: float = 0.15) -> tuple[float, float, float, float]:
    down_qtm = downsample(double_butter(data.qtm_cop[:, :2], 900), data.qtm_cop_ts, 15)
    length = min(down_qtm.shape[0], data.down_kinect_com.shape[0])
    o = max(int(length * cutoff), 1)

    down_qtm = down_qtm[:length][o:-o]
    com = data.down_kinect_com[:, :2][:length][o:-o]
    com_un = data.down_kinect_unfiltered_com[:, :2][:length][o:-o]

    corr = np.correlate(down_qtm[:, 0], com[:, 0])[0] + np.correlate(down_qtm[:, 1], com[:, 1])[0]
    corr_un = np.correlate(down_qtm[:, 0], com_un[:, 0])[0] + np.correlate(down_qtm[:, 1], com_un[:, 1])[0]

    diff = np.linalg.norm(down_qtm - com, axis=1)
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    diff_un = np.linalg.norm(down_qtm - com_un, axis=1)
    rmse_un = np.sqrt(np.mean(np.power(diff_un, 2)))

    return corr, corr_un, rmse, rmse_un


def find_best_measurement_error_factor_rmse(experiment_folder: Path, cutoff: float, experiment_type: str) -> tuple[Path, float]:
    directories = [element for element in experiment_folder.iterdir() if element.is_dir()]
    RMSEs = []
    all_RMSEs = []
    factors = []
    for directory in directories:
        data = load_processed_data(directory)

        #if float(data.config["measurement_error_factor"]) > 1.5:
        #    continue

        rmse = 0
        length = data.down_kinect_joints.shape[0]
        offset = max(int(length * cutoff), 1)

        if "constraint" in experiment_type:
            joints = [int(element) for element in [Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT]]
            for joint in joints:
                a = data.down_kinect_joints[:, joint, :]
                b = double_butter(data.down_kinect_unfiltered_joints[:, joint, :], N=2)

                # if  (0.3 < float(data.config["measurement_error_factor"]) < 0.35) or (1.0 < float(data.config["measurement_error_factor"]) < 1.05) or (5.00 < float(data.config["measurement_error_factor"]) < 5.05):
                '''
                plt.plot(data.down_kinect_ts, a[:, 2], label="kalman");
                plt.plot(data.down_kinect_ts, b[:, 2], label="butterworth");
                plt.plot(data.down_kinect_ts, data.down_kinect_unfiltered_joints[:, joint, 2], label="raw");
                plt.legend();
                plt.title(data.config["measurement_error_factor"]);
                plt.show();
                plt.cla();
                '''

                # Only take rmse in account for some% of the signal, prevent
                # weighting butterworth problems in the beginning and the end
                diff = np.linalg.norm(b[offset:-offset] - a[offset:-offset], axis=1)
                result = np.sqrt(np.mean(np.power(diff, 2)))

                rmse += result

                all_RMSEs.append(rmse)
        elif "cop" in experiment_type:
            a = data.down_kinect_com[:, :2]
            b = double_butter(data.down_kinect_unfiltered_com[:, :2], N=2)

            # Only take rmse in account for some% of the signal, prevent
            # weighting butterworth problems in the beginning and the end
            diff = np.linalg.norm(b[offset:-offset] - a[offset:-offset], axis=1)
            result = np.sqrt(np.mean(np.power(diff, 2)))

            rmse += result

            all_RMSEs.append(rmse)
        else:
            raise NotImplementedError(f"Invalid experiment_type: {experiment_type}")


        RMSEs.append(rmse)
        factors.append(data.config["measurement_error_factor"])

    RMSEs = np.array(RMSEs)

    plt.plot(np.array(factors), RMSEs, marker="X", ls="None")
    plt.xlabel("Measurement Error Factor")
    plt.ylabel("RMSE (Distance)")
    plt.legend()
    plt.title(f"Experiment: {os.path.basename(experiment_folder)} RMSE per measurement error factor")
    # plt.savefig(f"./results/experiments/factors_rmse_{experiment_type}_{os.path.basename(experiment_folder)}.pdf")
    plt.show()
    plt.cla()
    argmin = np.argmin(RMSEs)
    print(f"Min value: {RMSEs.min()}")
    return  directories[argmin], factors[argmin]

def find_best_measurement_error_factor_rmse_on_velocity(experiment_folder: Path, cutoff: float, experiment_type: str) -> tuple[Path, float]:
    """Find the best measurement error factor through comparing velocities."""
    directories = [element for element in experiment_folder.iterdir() if element.is_dir()]
    RMSEs = []
    factors = []
    once = True
    for directory in directories:
        data = load_processed_data(directory)

        length = data.down_kinect_joints.shape[0]
        o = int(length * cutoff)
        factor = float(data.config['measurement_error_factor'])

        rmse = 0
        for idx, joint in enumerate([Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT]):
            for i in range(3):
                a_vel = data.down_kinect_velocities[:, int(joint), i]
                b = double_butter(data.down_kinect_unfiltered_joints[:, int(joint), i], cutoff=2, once=True)
                c = data.down_kinect_unfiltered_joints[:, int(joint), i]

                b_vel = np.zeros_like(b)
                b_vel[1:-1] = (b[2:] - b[:-2]) / (2*(1./15.))
                b_vel[0] = (b[1] - b[0]) / (1./15.)
                b_vel[-1] = (b[-1] - b[-2]) / (1./15.)

                qtm_joints = double_butter(data.qtm_joints[:, idx, i], 150)

                butter_qtm_velocities = np.zeros_like(qtm_joints)
                butter_qtm_velocities[1:-1] = (qtm_joints[2:] - qtm_joints[:-2]) / (2*(1./150.))
                butter_qtm_velocities[0] = (qtm_joints[1] - qtm_joints[0]) / (1./150)
                butter_qtm_velocities[-1] = (qtm_joints[-1] - qtm_joints[-2]) / (1./150)
                d_vel = downsample(butter_qtm_velocities, data.qtm_ts, 15)

                # extract finite velocity
                # downsampled have 15 hz frequency
                '''
                if joint == Joint.WRIST_LEFT and once and i == 2 and 0 <= factor < 0.05:
                    once = False

                    c_vel = np.zeros_like(c)
                    c_vel[1:-1] = (c[2:] - c[:-2]) / (2*(1./15.))
                    c_vel[0] = (c[1] - c[0]) / (1./15.)
                    c_vel[-1] = (c[-1] - c[-2]) / (1./15.)

                    plt.plot(data.down_kinect_ts, a_vel, label="kalman vel")
                    plt.plot(data.down_kinect_ts, b_vel, label="butter fd vel")
                    # plt.plot(data.down_kinect_ts, c_vel, label="raw fd vel")
                    qtm_ts = np.arange(0, d_vel.shape[0]) * (1./15.)
                    plt.plot(qtm_ts, d_vel, label="qtm fd vel")
                    plt.legend()
                    plt.title(f"Error factor: {data.config['measurement_error_factor']}");
                    plt.show();
                '''

                error = np.sqrt(np.mean(np.power(a_vel[o:-o] - b_vel[o:-o], 2)))
                rmse += error

        RMSEs.append(rmse)
        factors.append(data.config["measurement_error_factor"])

    facts = np.array(factors)
    rmses = np.array(RMSEs)
    idx = np.argsort(facts)
    plt.plot(facts[idx][1:], rmses[idx][1:], marker="X", ls="None")
    plt.xlabel("Measurement Error Factor")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title(f"Experiment: {os.path.basename(experiment_folder)} RMSE per measurement error factor")
    plt.savefig(f"./results/experiments/factors_rmse_velocity_{experiment_type}_{os.path.basename(experiment_folder)}.pdf")
    plt.show()
    plt.cla()
    argmin = np.argmin(rmses)
    return directories[argmin], factors[argmin]

def find_best_measurement_error_factor_corr_on_velocity(experiment_folder: Path, cutoff: float, experiment_type: str) -> tuple[Path, float]:
    """Find the best measurement error factor through comparing velocities."""
    directories = [element for element in experiment_folder.iterdir() if element.is_dir()]
    correlations = []
    all_correlations = []
    factors = []
    once = True
    for directory in directories:
        data = load_processed_data(directory)

        length = data.down_kinect_joints.shape[0]
        o = int(length * cutoff)
        factor = float(data.config['measurement_error_factor'])

        correlation = 0
        for idx, joint in enumerate([Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT]):
            for i in range(3):
                a_vel = data.down_kinect_velocities[:, int(joint), i]
                b = double_butter(data.down_kinect_unfiltered_joints[:, int(joint), i], cutoff=2, once=True)
                c = data.down_kinect_unfiltered_joints[:, int(joint), i]

                b_vel = np.zeros_like(b)
                b_vel[1:-1] = (b[2:] - b[:-2]) / (2*(1./15.))
                b_vel[0] = (b[1] - b[0]) / (1./15.)
                b_vel[-1] = (b[-1] - b[-2]) / (1./15.)

                qtm_joints = double_butter(data.qtm_joints[:, idx, i], 150)

                butter_qtm_velocities = np.zeros_like(qtm_joints)
                butter_qtm_velocities[1:-1] = (qtm_joints[2:] - qtm_joints[:-2]) / (2*(1./150.))
                butter_qtm_velocities[0] = (qtm_joints[1] - qtm_joints[0]) / (1./150)
                butter_qtm_velocities[-1] = (qtm_joints[-1] - qtm_joints[-2]) / (1./150)
                d_vel = downsample(butter_qtm_velocities, data.qtm_ts, 15)

                # extract finite velocity
                # downsampled have 15 hz frequency
                if joint == Joint.WRIST_LEFT and once and i == 2 and 0 <= factor < 0.05:
                    once = False

                    c_vel = np.zeros_like(c)
                    c_vel[1:-1] = (c[2:] - c[:-2]) / (2*(1./15.))
                    c_vel[0] = (c[1] - c[0]) / (1./15.)
                    c_vel[-1] = (c[-1] - c[-2]) / (1./15.)

                    plt.plot(data.down_kinect_ts, a_vel, label="kalman vel")
                    plt.plot(data.down_kinect_ts, b_vel, label="butter fd vel")
                    # plt.plot(data.down_kinect_ts, c_vel, label="raw fd vel")
                    qtm_ts = np.arange(0, d_vel.shape[0]) * (1./15.)
                    plt.plot(qtm_ts, d_vel, label="qtm fd vel")
                    plt.legend()
                    plt.title(f"Error factor: {data.config['measurement_error_factor']}");
                    plt.show();

                corr = np.max(signal.correlate(a_vel[o:-o], b_vel[o:-o]))
                correlation += corr
                all_correlations.append(corr)

        correlations.append(correlation)
        factors.append(data.config["measurement_error_factor"])

    corrs = np.array(correlations)
    plt.plot(np.array(factors), corrs, marker="X", ls="None")
    plt.xlabel("Measurement Error Factor")
    plt.ylabel("Correlation")
    plt.legend()
    plt.title(f"Experiment: {os.path.basename(experiment_folder)} Correlation per measurement error factor")
    # plt.savefig(f"./results/experiments/factors_corr_{experiment_type}_{os.path.basename(experiment_folder)}.pdf")
    plt.show()
    argmax = np.argmax(corrs)
    return  directories[argmax], factors[argmax]


def find_best_measurement_error_factor_corr(experiment_folder: Path, cutoff: float, experiment_type: str) -> tuple[Path, float]:
    """Returns best factor path and factor."""
    directories = [element for element in experiment_folder.iterdir() if element.is_dir()]
    correlations = []
    all_correlations = []
    factors = []
    for directory in directories:
        data = load_processed_data(directory)

        length = data.down_kinect_joints.shape[0]
        o = int(length * cutoff)

        correlation = 0

        for joint in [Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT]:
            for i in range(3):
                a = data.down_kinect_joints[:, int(joint), i]
                b = double_butter(data.down_kinect_unfiltered_joints[:, int(joint), i])
                corr = np.correlate(a[o:-o], b[o:-o])[0]
                correlation += corr
                all_correlations.append(corr)

        print(correlation)
        correlations.append(correlation)
        factors.append(data.config["measurement_error_factor"])

    corrs = np.array(correlations)
    assert len(corrs) == len(directories)

    plt.plot(np.array(factors), corrs, marker="X", ls="None")
    plt.xlabel("Measurement Error Factor")
    plt.ylabel("Correlation")
    plt.legend()
    plt.title(f"Experiment: {os.path.basename(experiment_folder)} Correlation per measurement error factor")
    plt.savefig(f"./results/experiments/factors_corr_{experiment_type}_{os.path.basename(experiment_folder)}.pdf")
    plt.cla()
    argmax = np.argmax(corrs)
    return  directories[argmax], factors[argmax]

def compare_velocities(data: Data) -> None:
    """Compare the velocities of kalman filter, to finitie velocities of qtm."""

    qtm_joints = data.qtm_joints
    qtm_velocities = np.zeros_like(qtm_joints)
    qtm_velocities[1:-1, :] = (qtm_joints[2:, :] - qtm_joints[:-2, :]) / (2*(1./150.))
    qtm_velocities[0, :] = (qtm_joints[1, :] - qtm_joints[0, :]) / (1./150)
    qtm_velocities[-1, :] = (qtm_joints[-1, :] - qtm_joints[-2, :]) / (1./150)

    qtm_joints = double_butter(data.qtm_joints, 150)

    butter_qtm_velocities = np.zeros_like(qtm_joints)
    butter_qtm_velocities[1:-1, :] = (qtm_joints[2:, :] - qtm_joints[:-2, :]) / (2*(1./150.))
    butter_qtm_velocities[0, :] = (qtm_joints[1, :] - qtm_joints[0, :]) / (1./150)
    butter_qtm_velocities[-1, :] = (qtm_joints[-1, :] - qtm_joints[-2, :]) / (1./150)

    b = double_butter(data.down_kinect_unfiltered_joints[:, int(Joint.ELBOW_LEFT), 2], cutoff=2, once=True)

    b_vel = np.zeros_like(b)
    b_vel[1:-1] = (b[2:] - b[:-2]) / (2*(1./15.))
    b_vel[0] = (b[1] - b[0]) / (1./15.)
    b_vel[-1] = (b[-1] - b[-2]) / (1./15.)


    kinect_velocities = np.zeros((data.down_kinect_joints.shape[0], 3, 3))
    kinect_velocities[:, 0, :] = data.down_kinect_velocities[:, int(Joint.SHOULDER_LEFT), :]
    kinect_velocities[:, 1, :] = data.down_kinect_velocities[:, int(Joint.ELBOW_LEFT), :]
    kinect_velocities[:, 2, :] = data.down_kinect_velocities[:, int(Joint.WRIST_LEFT), :]
    plt.plot(np.arange(1, qtm_velocities.shape[0]+1) * (1./150.), qtm_velocities[:, 1, 2], label="qtm")
    plt.plot(np.arange(1, qtm_velocities.shape[0]+1) * (1./150.), butter_qtm_velocities[:, 1, 2], label="butter qtm")
    plt.plot(data.down_kinect_ts, kinect_velocities[:, 1, 2], label="kinect")
    plt.plot(data.down_kinect_ts, b_vel, label="butter raw vel")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Velocity of Joint ? - Axis ?")
    plt.legend()
    plt.show()
    plt.cla()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder")
    parser.add_argument("-t", "--experiment-type", dest="experiment_type", choices=["cop", "cop-wide", "constraint", "constraint-fast"])
    parser.add_argument("-x", "--early-exit", dest="early_exit", action="store_true")

    args = parser.parse_args()

    cutoff = 0.01
    # path, factor = find_best_measurement_error_factor_rmse(Path(args.experiment_folder), cutoff, args.experiment_type)
    # path, factor = find_best_measurement_error_factor_corr_on_velocity(Path(args.experiment_folder), cutoff, args.experiment_type)
    path, factor = find_best_measurement_error_factor_rmse_on_velocity(Path(args.experiment_folder), cutoff, args.experiment_type)


    print(path)
    print(f"factor: {factor}")
    data = load_processed_data(path)

    result = None
    if args.experiment_type in ["cop", "cop-wide"]:
        result = compare_qtm_cop_kinect_cop(data, cutoff)
    else:
        result = compare_qtm_joints_kinect_joints(data, cutoff)

    print(result)

    # compare_velocities(load_processed_data(Path(args.experiment_folder) / "6"))
    compare_velocities(data)

    if args.early_exit:
        return

    if args.experiment_type in ["cop", "cop-wide"]:
        plt.cla()
        o = int(data.down_kinect_com.shape[0] * cutoff)
        _, ax = plt.subplots(1, 2)
        ax[0].plot(data.down_kinect_ts[o:-o], data.down_kinect_com[:, 0][o:-o], label="Kalman Filtered")
        ax[0].plot(data.down_kinect_ts[o:-o], double_butter(data.down_kinect_unfiltered_com[:, 0])[o:-o], label="Double Butterworth Filtered")
        ax[0].plot(data.down_kinect_ts[o:-o], data.down_kinect_unfiltered_com[:, 0][o:-o], label="Raw Data")
        ax[0].set_title("X Axis")
        ax[0].legend()

        ax[1].plot(data.down_kinect_ts[o:-o], data.down_kinect_com[:, 1][o:-o], label="Kalman Filtered")
        ax[1].plot(data.down_kinect_ts[o:-o], double_butter(data.down_kinect_unfiltered_com[:, 1])[o:-o], label="Double Butterworth Filtered")
        ax[1].plot(data.down_kinect_ts[o:-o], data.down_kinect_unfiltered_com[:, 1][o:-o], label="Raw Data")
        ax[1].set_title("Y Axis")
        ax[1].legend()

        plt.show()


    else:
        # Manual check to see if the butterworth artifacts have been cutoff
        o = int(data.down_kinect_joints.shape[0] * cutoff)
        plt.plot(data.down_kinect_ts[o:-o], data.down_kinect_joints[:, int(Joint.ELBOW_LEFT), 2][o:-o], label="Kalman Filtered")
        plt.plot(data.down_kinect_ts[o:-o], double_butter(data.down_kinect_unfiltered_joints[:, int(Joint.ELBOW_LEFT), 2])[o:-o], label="Double Butterworth Filtered")
        plt.plot(data.down_kinect_ts[o:-o], data.down_kinect_unfiltered_joints[:, int(Joint.ELBOW_LEFT), 2][o:-o], label="Raw Data")
        plt.legend()
        plt.show()

    print(f"experiment type: {args.experiment_type}")
    print(factor)


def plot_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder")

    args = parser.parse_args()

    data = load_processed_data(Path(args.experiment_folder))
    plt.plot(data.down_kinect_ts, data.down_kinect_com[:, 0], label="kinect_com")
    plt.plot(data.down_kinect_ts, double_butter(data.down_kinect_com[:, 0]), label="butter kinect_com")
    plt.plot(data.qtm_cop_ts, data.qtm_cop[:, 0], label="qtm cop")
    plt.plot(data.qtm_cop_ts, double_butter(data.qtm_cop[:, 0], 900), label="butter qtm cop")
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

    out = double_butter(data.qtm_cop, 900)
    assert np.all(out[:, 0] == double_butter(data.qtm_cop[:, 0], 900))

    result = downsample(data.qtm_cop[:, 0], data.qtm_cop_ts, 450)
    diff = result - data.down_qtm_cop[:, 0]
    assert_allclose(diff, 0)

    result = downsample(data.qtm_cop, data.qtm_cop_ts, 450)
    diff = result - data.down_qtm_cop
    assert_allclose(diff, 0)

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
    main()
    # test()
