from __future__ import annotations
import time
from typing import Optional
from pyCompare import blandAltman
from pprint import pprint as pp
import os
import json
import numpy as np
from numpy.testing import assert_allclose
import argparse
from dataclasses import dataclass, field
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from enum import IntEnum
from scipy import signal
from scipy.spatial import distance
import numba
import dtw
from tqdm import tqdm
import numba
from multiprocessing.pool import ThreadPool
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

params = {'text.usetex' : True,
          'font.size' : 16,
          'font.family' : 'lmodern'
          }
plt.rcParams.update(params)

SHOW = False

def wt(text: str) -> str:
    return rf"\begin{{tabular}}[c]{{@{{}}l@{{}}}}{text}\end{{tabular}}"

def corr_shift_trim3d(a: np.ndarray, b: np.ndarray, idx: int = 2) -> tuple[np.ndarray, np.ndarray]:
        x, y =  a[:, idx], b[:, idx]
        off = np.argmax(signal.correlate(x, y)) - len(a) + 1
        if off < 0:
            b = b[abs(off):]
            length = min(a.shape[0], b.shape[0])
            return a[:length], b[:length]
        elif off > 0:
            a = a[abs(off):]
            length = min(a.shape[0], b.shape[0])
            return a[:length], b[:length]
        return a, b


def bland_altman(data1: np.ndarray, data2: np.ndarray, path: str) -> None:
    count = bland_altman_outlier_count(data1, data2)
    blandAltman(
        data1,
        data2,
        limitOfAgreement=1.96,
        confidenceInterval=95,
        confidenceIntervalMethod="approximate",
        detrend=None,
        percentage=False,
        figureFormat="pdf",
        title=f"Outlier: {count}",
        savePath=path
    )

def bland_altman_outlier_count(data1: np.ndarray, data2: np.ndarray) -> int:
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   # Difference between data1 and data2
    md = np.mean(diff)                   # Mean of the difference
    sd = np.std(diff, axis=0)
    return np.sum(diff < md-sd*1.96) + np.sum(diff > md+sd*1.96)

FILTER_NAME = ""

# Adapted from https://github.com/cjekel/similarity_measures to support jit
# compilation using numba
def frechet_dist(exp_data, num_data, p=2):
    r"""
    Compute the discrete Frechet distance

    Compute the Discrete Frechet Distance between two N-D curves according to
    [1]_. The Frechet distance has been defined as the walking dog problem.
    From Wikipedia: "In mathematics, the Frechet distance is a measure of
    similarity between curves that takes into account the location and
    ordering of the points along the curves. It is named after Maurice Frechet.
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance

    Parameters
    ----------
    exp_data : array_like
        Curve from your experimental data. exp_data is of (M, N) shape, where
        M is the number of data points, and N is the number of dimmensions
    num_data : array_like
        Curve from your numerical data. num_data is of (P, N) shape, where P
        is the number of data points, and N is the number of dimmensions
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use. Default is p=2 (Eculidean).
        The manhattan distance is p=1.

    Returns
    -------
    df : float
        discrete Frechet distance

    References
    ----------
    .. [1] Thomas Eiter and Heikki Mannila. Computing discrete Frechet
        distance. Technical report, 1994.
        http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.937&rep=rep1&type=pdf

    Notes
    -----
    Your x locations of data points should be exp_data[:, 0], and the y
    locations of the data points should be exp_data[:, 1]. Same for num_data.

    Thanks to Arbel Amir for the issue, and Sen ZHANG for the iterative code
    https://github.com/cjekel/similarity_measures/issues/6

    Examples
    --------
    >>> # Generate random experimental data
    >>> x = np.random.random(100)
    >>> y = np.random.random(100)
    >>> exp_data = np.zeros((100, 2))
    >>> exp_data[:, 0] = x
    >>> exp_data[:, 1] = y
    >>> # Generate random numerical data
    >>> x = np.random.random(100)
    >>> y = np.random.random(100)
    >>> num_data = np.zeros((100, 2))
    >>> num_data[:, 0] = x
    >>> num_data[:, 1] = y
    >>> df = frechet_dist(exp_data, num_data)

    """
    n = len(exp_data)
    m = len(num_data)
    c = distance.cdist(exp_data, num_data, metric='minkowski', p=p)
    ca = np.ones((n, m))
    ca = np.multiply(ca, -1)
    ca[0, 0] = c[0, 0]
    return _frechet_dist(ca, c, n ,m)

@numba.jit(nopython=True)
def _frechet_dist(ca, c, n, m):
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], c[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], c[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                           c[i, j])
    return ca[n-1, m-1]

def central_diff(data: np.ndarray, frequency: float):
    data_vel = np.zeros_like(data)
    data_vel[1:-1] = (data[2:] - data[:-2]) / (2*(1./frequency))
    data_vel[0] = (data[1] - data[0]) / (1./frequency)
    data_vel[-1] = (data[-1] - data[-2]) / (1./frequency)
    return data_vel

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
class DetermineFactorRecord:
    experiment_name: str
    rmse: float
    dtw: float
    dfd: float
    pcc: float


@dataclass
class DetermineFactorGenerateTable:
    filter_type: str
    eval_type: str
    records: list[DetermineFactorRecord] = field(default_factory=list)
    vel: bool = False

    def append(self, record: DetermineFactorRecord) -> None:
        self.records.append(record)

    def mean(self) -> DetermineFactorRecord:
        mean_record = DetermineFactorRecord("mean", 0, 0, 0, 0)
        for record in self.records:
            mean_record.rmse += record.rmse
            mean_record.dtw += record.dtw
            mean_record.dfd += record.dfd
            mean_record.pcc += record.pcc

        mean_record.rmse = mean_record.rmse / len(self.records)
        mean_record.dtw = mean_record.dtw / len(self.records)
        mean_record.dfd = mean_record.dfd / len(self.records)
        mean_record.pcc = mean_record.pcc / len(self.records)
        return mean_record

    def generate_table(self, path: Path) -> None:
        begin = r'''
\begin{table}[]
\begin{center}
\begin{tabular}{|l|l|l|l|l|}
\hline
\rowcolor[HTML]{C0C0C0}
\textbf{Experiment Name} & \textbf{RMSE} & \textbf{DTW} & \textbf{DFD} & \textbf{PCC} \\ \hline
'''
        content = ""
        for record in self.records:
            ex_name = record.experiment_name.replace("_", "\\_")
            content += rf"{map_ex_name(ex_name)}                   & {round(record.rmse, 1)}            & {round(record.dtw, 1)}           & {round(record.dfd, 1)}           & {round(record.pcc, 1)}           \\ \hline"
            content += "\n"
        end = rf'''
\end{{tabular}}
\end{{center}}
\caption{{Optimal $\lambda$ for \{self.filter_type}\ Evaluated on {self.eval_type}}}
\label{{tab:{"vel-" if self.vel else ""}determine-factor-{self.filter_type}}}
\end{{table}}
'''

        table_str = begin + content + end
        print(table_str)
        with path.open(mode="w+", encoding="UTF-8") as output_file:
            output_file.write(table_str)

@dataclass
class GenericTableGeneration:
    header_names: list[str]
    title: str
    ref: str
    records: list[list[str]] = field(default_factory=list)
    rotation: int = 0

    def append(self, record: list[str]) -> None:
        self.records.append(record)

    def generate_table(self, path: Path) -> None:
        border = "|".join(["l"] * len(self.header_names))
        border = "|" + border + "|"
        begin = rf'''
\begin{{table}}[]
\begin{{center}}
\begin{{adjustbox}}{{angle={self.rotation}}}
\begin{{tabular}}{{{border}}}
\hline
\rowcolor[HTML]{{C0C0C0}}
'''
        escape_header = [wt(name) if r"\\" in name else name for name in self.header_names]
        header = " & ".join([rf"\textbf{{{name}}}" for name in escape_header])
        header += r" \\ \hline"
        begin += "\n"
        begin += header
        begin += "\n"
        content = ""
        for record in self.records:
            escape_record = [wt(item) if r"\\" in item else item for item in record]
            content += " & ".join([rf"{item}" for item in escape_record])
            content += r" \\ \hline"
            content += "\n"
        end = rf'''
\end{{tabular}}
\end{{adjustbox}}
\end{{center}}
\caption{{{self.title}}}
\label{{tab:{self.ref}}}
\end{{table}}
'''

        table_str = begin + content + end
        print(table_str)
        with path.open(mode="w+", encoding="UTF-8") as output_file:
            output_file.write(table_str)


@dataclass
class TheiaData:
    config: dict[str, str]
    down_kinect_com: np.ndarray
    down_kinect_com_velocities: np.ndarray
    down_kinect_joints: np.ndarray
    down_kinect_predictions: np.ndarray
    down_kinect_ts: np.ndarray
    down_kinect_unfiltered_com: np.ndarray
    down_kinect_unfiltered_joints: np.ndarray
    down_kinect_velocities: np.ndarray
    down_theia_tensor: np.ndarray
    kinect_com: np.ndarray
    kinect_com_velocities: np.ndarray
    kinect_joints: np.ndarray
    kinect_predictions: np.ndarray
    kinect_ts: np.ndarray
    kinect_unfiltered_com: np.ndarray
    kinect_unfiltered_joints: np.ndarray
    kinect_velocities: np.ndarray
    theia_tensor: np.ndarray

    length_theia_joints: Optional[int] = field(init=False, default=None)

    @property
    def min_joint_length_at_15hz(self) -> int:
        if not self.length_theia_joints:
            length_theia_tensor = self.theia_tensor.shape[0]
            self.length_theia_joints = downsample(self.theia_tensor, np.arange(length_theia_tensor) * (1./120.), 15).shape[0]
        return min(self.down_kinect_joints.shape[0], self.down_kinect_unfiltered_joints.shape[0], self.length_theia_joints)

    @property
    def down_kinect_xcom(self) -> np.ndarray:
        """Returns kinect xcom with 15 hz."""
        g = 9.81
        w_0 = np.sqrt(g / self.down_kinect_com[:, 2])
        len_w_0 = len(w_0)
        w_0 = np.repeat(w_0, 3).reshape(len_w_0, 3)
        return self.down_kinect_com + (self.down_kinect_com_velocities / w_0)

    @property
    def down_kinect_unfiltered_xcom(self) -> np.ndarray:
        """Returns kinect xcom with 15 hz."""
        g = 9.81
        w_0 = np.sqrt(g / self.down_kinect_com[:, 2])
        len_w_0 = len(w_0)
        w_0 = np.repeat(w_0, 3).reshape(len_w_0, 3)
        return self.down_kinect_com + (self.down_kinect_com_velocities / w_0)


    @property
    def down_theia_ts(self) -> np.ndarray:
        return np.arange(self.down_theia_tensor.shape[0]) * (1./30.)

    @property
    def down_theia_xcom_15_hz(self) -> np.ndarray:
        return downsample(self.down_theia_xcom, self.down_theia_ts, 15)

    @property
    def down_theia_xcom(self) -> np.ndarray:
        """Returns theia xcom with 120 hz."""
        g = 9.81
        w_0 = np.sqrt(g / self.down_theia_tensor[:, int(TheiaJoint.COM), 2])
        len_w_0 = len(w_0)
        w_0 = np.repeat(w_0, 3).reshape(len_w_0, 3)
        return self.down_theia_tensor[:, int(TheiaJoint.COM), :] + (self.down_theia_tensor[:, int(TheiaJoint.COM_VEL), :] / w_0)


@dataclass
class Data:
    down_kinect_com: np.ndarray
    down_kinect_com_velocities: np.ndarray
    down_kinect_joints: np.ndarray
    down_kinect_predictions: np.ndarray
    down_kinect_ts: np.ndarray
    down_kinect_unfiltered_com: np.ndarray
    down_kinect_unfiltered_joints: np.ndarray
    down_kinect_velocities: np.ndarray
    down_qtm_cop: np.ndarray
    down_qtm_cop_ts: np.ndarray
    down_qtm_joints: np.ndarray
    down_qtm_ts: np.ndarray
    kinect_com: np.ndarray
    kinect_com_velocities: np.ndarray
    kinect_joints: np.ndarray
    kinect_predictions: np.ndarray
    kinect_ts: np.ndarray
    kinect_unfiltered_com: np.ndarray
    kinect_unfiltered_joints: np.ndarray
    kinect_velocities: np.ndarray
    qtm_cop: np.ndarray
    qtm_cop_ts: np.ndarray
    qtm_joints: np.ndarray
    qtm_ts: np.ndarray
    config: dict[str, str]

    length_qtm_joints: Optional[int] = field(init=False, default=None)

    @property
    def min_joint_length_at_15hz(self) -> int:
        if not length_qtm_joints:
            length_qtm_joints = downsample(qtm_joints, qtm_ts, 15).shape[0]
        return min(self.down_kinect_joints.shape[0], self.down_kinect_unfiltered_joints.shape[0], length_qtm_joints)


_THEIA_JOINTS = [
    "NECK",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "COM",
    "COM_VEL",
]
TheiaJoint = IntEnum("TheiaJoint", _THEIA_JOINTS, start = 0)

def theiaj2str(idx: int) -> str:
    return _THEIA_JOINTS[idx]

_JOINTS = [
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
]
Joint = IntEnum("Joint", _JOINTS, start = 0)

def j2str(idx: int) -> str:
    return _JOINTS[idx]

FILTER_TYPES = ["ConstrainedSkeletonFilter", "SkeletonFilter", "SimpleConstrainedSkeletonFilter", "SimpleSkeletonFilter"]
SHORT_FILTER_TYPES = ["CSF", "SF", "SCSF", "SSF"]
FILTER_NAME_MAP = {
    "ConstrainedSkeletonFilter": "CSF",
    "SkeletonFilter": "SF",
    "SimpleConstrainedSkeletonFilter": "SCSF",
    "SimpleSkeletonFilter": "SSF",
}

def short_name(long_name: str) -> str:
    return FILTER_NAME_MAP[long_name]

YLABELS = ["RMSE [m]", None, None, None]
METRIC_NAMES = ["RMSE", "DTW", "DFD", "PCC"]

EX_NAMES = [
    "s10001",
    "s10003",
    "s10002",
    "s10004",
    "s30001",
    "s30004",
    "s30005",
    "s30002",
    "s30003",
    "s30006",
]

EX_NAMES_WITH_BETTER = [
    "s10001",
    "s10003",
    "s10002",
    "s10004",
    "s30001",
    "s30004",
    "s30005",
    "s30002",
    "s30003",
    "s30006",
    "b_s30001",
    "b_s30004",
    "b_s30005",
    "b_s30003",
    "b_s30002",
    "b_s30006",
]

THEIA_EX_NAMES = [
    "s30001",
    "s30004",
    "s30005",
    "s30003",
    "s30002",
    "s30006",
]

EX_NAMES_WITH_BETTER_AND_SMOOTHED = [
    "s10001",
    "s10003",
    "s10002",
    "s10004",
    "s30001",
    "s30003",
    "s30002",
    "s30004",
    "s30005",
    "s30006",
    "b_s30001",
    "b_s30003",
    "b_s30002",
    "b_s30004",
    "b_s30005",
    "b_s30006",
    "sb_s30001",
    "sb_s30002",
    "sb_s30003",
    "sb_s30004",
    "sb_s30005",
    "sb_s30006",
]

EX_NAME_MAPPING = {
        "s10001": "1a@2 +NN",
        "s10002": "1b@2 +NN",
        "s10003": "1a@3 +NN",
        "s10004": "1b@3 +NN",
        "s30001": "2b@1",
        "s30003": "2a@3",
        "s30002": "2b@3",
        "s30004": "2a@2",
        "s30005": "2b@2",
        "s30006": "3@3",
        "b_s30001": "2b@1 +NN",
        "b_s30004": "2a@2 +NN",
        "b_s30005": "2b@2 +NN",
        "b_s30003": "2a@3 +NN",
        "b_s30002": "2b@3 +NN",
        "b_s30006": "3@3 +NN",
        "sb_s30001": "2b@1 +S(0.5) +NN",
        "sb_s30004": "2a@2 +S(0.5) +NN",
        "sb_s30005": "2b@2 +S(0.5) +NN",
        "sb_s30003": "2a@3 +S(0.5) +NN",
        "sb_s30002": "2b@3 +S(0.5) +NN",
        "sb_s30006": "3@3 +S(0.5) +NN",
        "b\\_s30001": "2b@1 +NN",
        "b\\_s30004": "2a@2 +NN",
        "b\\_s30005": "2b@2 +NN",
        "b\\_s30003": "2a@3 +NN",
        "b\\_s30002": "2b@3 +NN",
        "b\\_s30006": "3@3 +NN",
        "sb\\_s30001": "2b@1 +S(0.5) +NN",
        "sb\\_s30004": "2a@2 +S(0.5) +NN",
        "sb\\_s30005": "2b@2 +S(0.5) +NN",
        "sb\\_s30003": "2a@3 +S(0.5) +NN",
        "sb\\_s30002": "2b@3 +S(0.5) +NN",
        "sb\\_s30006": "3@3 +S(0.5) +NN",
}

def map_ex_name(ex_name: str) -> str:
    return EX_NAME_MAPPING[ex_name]

MATCHING_JOINTS = [
    (Joint.SHOULDER_LEFT, TheiaJoint.SHOULDER_LEFT, "Left Shoulder"),
    (Joint.ELBOW_LEFT, TheiaJoint.ELBOW_LEFT, "Left Elbow"),
    (Joint.WRIST_LEFT, TheiaJoint.WRIST_LEFT, "Left Wrist"),
    (Joint.ANKLE_LEFT, TheiaJoint.ANKLE_LEFT, "Left Ankle"),
    (Joint.KNEE_LEFT, TheiaJoint.KNEE_LEFT, "Left Knee"),
    (Joint.HIP_LEFT, TheiaJoint.HIP_LEFT, "Left Hip"),
    (Joint.FOOT_LEFT, TheiaJoint.FOOT_LEFT, "Left Foot"),
    (Joint.SHOULDER_RIGHT, TheiaJoint.SHOULDER_RIGHT, "Right Shoulder"),
    (Joint.ELBOW_RIGHT, TheiaJoint.ELBOW_RIGHT, "Right Elbow"),
    (Joint.WRIST_RIGHT, TheiaJoint.WRIST_RIGHT, "Right Wrist"),
    (Joint.ANKLE_RIGHT, TheiaJoint.ANKLE_RIGHT, "Right Ankle"),
    (Joint.KNEE_RIGHT, TheiaJoint.KNEE_RIGHT, "Right Knee"),
    (Joint.HIP_RIGHT, TheiaJoint.HIP_RIGHT, "Right Hip"),
    (Joint.FOOT_RIGHT, TheiaJoint.FOOT_RIGHT, "Right Foot"),
    (Joint.NECK, TheiaJoint.NECK, "Neck"),
]

JOINT_SEGMENTS = [
    ([Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT], [TheiaJoint.SHOULDER_LEFT, TheiaJoint.ELBOW_LEFT, TheiaJoint.WRIST_LEFT], "UP_LEFT"),
    ([Joint.SHOULDER_RIGHT, Joint.ELBOW_RIGHT, Joint.WRIST_RIGHT], [TheiaJoint.SHOULDER_RIGHT, TheiaJoint.ELBOW_RIGHT, TheiaJoint.WRIST_RIGHT], "UP_RIGHT"),
    ([Joint.ANKLE_LEFT, Joint.KNEE_LEFT, Joint.HIP_LEFT], [TheiaJoint.ANKLE_LEFT, TheiaJoint.KNEE_LEFT, TheiaJoint.HIP_LEFT], "DOWN_LEFT"),
    ([Joint.ANKLE_RIGHT, Joint.KNEE_RIGHT, Joint.HIP_RIGHT], [TheiaJoint.ANKLE_RIGHT, TheiaJoint.KNEE_RIGHT, TheiaJoint.HIP_RIGHT], "DOWN_RIGHT"),
]

SEGMENT_NAME_TO_A_B_NAME = {
    "UP_LEFT": ("Upper Arm Left", "Lower Arm Left"),
    "UP_RIGHT": ("Upper Arm Right", "Lower Arm Right"),
    "DOWN_LEFT": ("Thigh Left", "Shank Left"),
    "DOWN_RIGHT": ("Thigh Right", "Shank Right"),
}

A_B_NAMES = [
    "Upper Arm Left", "Lower Arm Left",
    "Upper Arm Right", "Lower Arm Right",
    "Thigh Left", "Shank Left",
    "Thigh Right", "Shank Right"
]

def segment_name_to_a_b_name(name: str) -> tuple[str, str]:
    return SEGMENT_NAME_TO_A_B_NAME[name]



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

def cond_load_data(path: Path) -> Data | TheiaData:
    if (path / "theia_tensor.npy").exists():
        return load_processed_theia_data(path)
    else:
        return load_processed_data(path)

def load_processed_theia_data(path: Path) -> TheiaData:
    return TheiaData(
        json.load((path / "config.json").open(mode="r", encoding="UTF-8")),
        np.load(path / "down_kinect_com.npy"),
        np.load(path / "down_kinect_com_velocities.npy"),
        np.load(path / "down_kinect_joints.npy"),
        np.load(path / "down_kinect_predictions.npy"),
        np.load(path / "down_kinect_ts.npy"),
        np.load(path / "down_kinect_unfiltered_com.npy"),
        np.load(path / "down_kinect_unfiltered_joints.npy"),
        np.load(path / "down_kinect_velocities.npy"),
        np.load(path / "down_theia_tensor.npy"),
        np.load(path / "kinect_com.npy"),
        np.load(path / "kinect_com_velocities.npy"),
        np.load(path / "kinect_joints.npy"),
        np.load(path / "kinect_predictions.npy"),
        np.load(path / "kinect_ts.npy"),
        np.load(path / "kinect_unfiltered_com.npy"),
        np.load(path / "kinect_unfiltered_joints.npy"),
        np.load(path / "kinect_velocities.npy"),
        np.load(path / "theia_tensor.npy"),
    )


def load_processed_data(path: Path) -> Data:
    return Data(
        np.load(path / "down_kinect_com.npy"),
        np.load(path / "down_kinect_com_velocities.npy"),
        np.load(path / "down_kinect_joints.npy"),
        np.load(path / "down_kinect_predictions.npy"),
        np.load(path / "down_kinect_ts.npy"),
        np.load(path / "down_kinect_unfiltered_com.npy"),
        np.load(path / "down_kinect_unfiltered_joints.npy"),
        np.load(path / "down_kinect_velocities.npy"),
        np.load(path / "down_qtm_cop.npy"),
        np.load(path / "down_qtm_cop_ts.npy"),
        np.load(path / "down_qtm_joints.npy"),
        np.load(path / "down_qtm_ts.npy"),
        np.load(path / "kinect_com.npy"),
        np.load(path / "kinect_com_velocities.npy"),
        np.load(path / "kinect_joints.npy"),
        np.load(path / "kinect_predictions.npy"),
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


def plot_velocities_for_different_factors(ex_name: str, factors: list[float], datas: list[Data], cutoff: float, plotsuffix: str = "") -> None:
    kinect_joints = [Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT]

    data = datas[0]

    qtm_ts = data.qtm_ts

    labels = ["X", "Y", "Z"]
    for i, label in enumerate(labels):
        for qtm_idx, joint in enumerate(kinect_joints):
            data = datas[0]
            qtm_line = double_butter(data.qtm_joints[:, qtm_idx, i], sample_frequency=150)
            q_vel = np.zeros_like(qtm_line)
            q_vel[1:-1] = (qtm_line[2:] - qtm_line[:-2]) / (2*(1./150.))
            q_vel[0] = (qtm_line[1] - qtm_line[0]) / (1./150)
            q_vel[-1] = (qtm_line[-1] - qtm_line[-2]) / (1./150)

            q_vel_function = central_diff(qtm_line, 150)
            assert_allclose(q_vel, q_vel_function)

            plt.plot(qtm_ts, q_vel, label=f"Qualisys", alpha=0.8, marker="x", markevery=500)
            for factor, data in zip(factors, datas):
                kinect_ts = data.kinect_ts
                line = data.kinect_velocities[:, int(joint), i]
                plt.plot(kinect_ts, line, label=f"Kalman Kinect with Factor: {factor}", markevery=50, marker=".", alpha=0.3)

            plt.legend()
            plt.xlabel("Time [s]")
            plt.ylabel(f"{label} Axis [m]")

            plt.title(rf"Velocity of {j2str(joint)} with different $\lambda$")
            plt.savefig(f"./results/experiments/{FILTER_NAME}/joint_velocities/{j2str(joint)}_axis_{label}_{ex_name}_{plotsuffix}.pdf", bbox_inches="tight")
            plt.cla()


def plot_joints_for_different_factors(ex_name: str, factors: list[float], datas: list[Data], cutoff: float, plotsuffix: str = "") -> None:
    kinect_joints = [Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT]

    data = datas[0]

    qtm_ts = data.qtm_ts

    labels = ["X", "Y", "Z"]
    for i, label in enumerate(labels):
        for qtm_idx, joint in enumerate(kinect_joints):
            data = datas[0]
            qtm_line = double_butter(data.qtm_joints[:, qtm_idx, i], sample_frequency=150)
            plt.plot(qtm_ts, qtm_line, label=f"Qualisys", alpha=0.3)
            for factor, data in zip(factors, datas):
                kinect_ts = data.kinect_ts
                line = data.kinect_joints[:, int(joint), i]
                plt.plot(kinect_ts, line, label=f"Kalman Kinect with Factor: {factor}", markevery=50, marker=".", alpha=0.3)


            plt.legend()
            plt.xlabel("Time [s]")
            plt.ylabel(f"{label} Axis [m]")

            plt.title(rf"Trajectiories of {j2str(joint)} with different $\lambda$")
            plt.savefig(f"./results/experiments/{FILTER_NAME}/joint_trajectories/{j2str(joint)}_axis_{label}_{ex_name}_{plotsuffix}.pdf", bbox_inches="tight")
            plt.cla()

def plot_cop_x_y_for_different_factors(ex_name: str, factors: list[float], datas: list[Data], cutoff: float, plotsuffix: str = "") -> None:

    data = datas[0]

    qtm_ts = data.qtm_cop_ts

    data = datas[0]
    for i, label in enumerate(["X", "Y"]):
        qtm_line = double_butter(data.qtm_cop[:, i], sample_frequency=900)
        plt.plot(qtm_ts, qtm_line, label=f"Qualisys", alpha=0.3)

        for factor, data in zip(factors, datas):
            # For all three axis
            kinect_ts = data.kinect_ts
            line = data.kinect_com[:, i]
            plt.plot(kinect_ts, line, label=f"Kalman Kinect with Factor: {factor}", markevery=50, marker=".", alpha=0.3)

        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel(f"{label} Axis [m]")

        plt.suptitle(rf"Trajectiories of COM and COP with different $\lambda$")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/cop_trajectories/{ex_name}_axis_{label}_{plotsuffix}.pdf", bbox_inches="tight")
        plt.cla()



def plot_constrained_segment_joint_length_change(ex_name: str, data: Data | TheiaData, cutoff: float) -> list[tuple]:
    segment_a = [int(element) for element in [Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT]]
    segment_b = [int(element) for element in [Joint.SHOULDER_RIGHT, Joint.ELBOW_RIGHT, Joint.WRIST_RIGHT]]
    segment_c = [int(element) for element in [Joint.HIP_LEFT, Joint.KNEE_LEFT, Joint.ANKLE_LEFT]]
    segment_d = [int(element) for element in [Joint.HIP_RIGHT, Joint.KNEE_RIGHT, Joint.ANKLE_RIGHT]]

    is_theia = isinstance(data, TheiaData)
    records = []
    for kinect_joints, theia_joints, name in JOINT_SEGMENTS:
        segment_name = "_".join([str(i) for i in kinect_joints]) + "_" + name

        length = min(data.kinect_joints.shape[0], data.kinect_unfiltered_joints.shape[0])
        o = max(int(length * cutoff), 1)

        ts = data.kinect_ts[o:-o]

        a_theia, b_theia = None, None

        shoulder = data.kinect_joints[:, kinect_joints[0], :][:length][o:-o]
        shoulder_un = data.kinect_unfiltered_joints[:, kinect_joints[0], :][:length][o:-o]
        if is_theia:
            shoulder_theia = data.theia_tensor[:, theia_joints[0], :][:length][o:-o]
        butter_shoulder_un = double_butter(data.kinect_unfiltered_joints[:, kinect_joints[0], :])[:length][o:-o]

        elbow = data.kinect_joints[:, kinect_joints[1], :][:length][o:-o]
        elbow_un = data.kinect_unfiltered_joints[:, kinect_joints[1], :][:length][o:-o]
        if is_theia:
            elbow_theia = data.theia_tensor[:, theia_joints[1], :][:length][o:-o]
        butter_elbow_un = double_butter(data.kinect_unfiltered_joints[:, kinect_joints[1], :])[:length][o:-o]

        wrist = data.kinect_joints[:, kinect_joints[2], :][:length][o:-o]
        wrist_un = data.kinect_unfiltered_joints[:, kinect_joints[2], :][:length][o:-o]
        if is_theia:
            wrist_theia = data.theia_tensor[:, theia_joints[2], :][:length][o:-o]
        butter_wrist_un = double_butter(data.kinect_unfiltered_joints[:, kinect_joints[2], :])[:length][o:-o]

        a = np.linalg.norm(shoulder - elbow, axis=1)
        b = np.linalg.norm(elbow - wrist, axis=1)

        a_un = np.linalg.norm(shoulder_un - elbow_un, axis=1)
        b_un = np.linalg.norm(elbow_un - wrist_un, axis=1)

        if is_theia:
            a_theia = np.linalg.norm(shoulder_theia - elbow_theia, axis=1)
            b_theia = np.linalg.norm(elbow_theia - wrist_theia, axis=1)

        butter_a_un = np.linalg.norm(butter_shoulder_un - butter_elbow_un, axis=1)
        butter_b_un = np.linalg.norm(butter_elbow_un - butter_wrist_un, axis=1)

        print(f"Segment {segment_name}")
        print("Kalman")
        print(f"a mean: {a.mean()}, b mean: {b.mean()}, a.var: {a.var()}, b.var: {b.var()}")
        print(f"a_un mean: {a_un.mean()}, b_un mean: {b_un.mean()}, a_un.var: {a_un.var()}, b_un.var: {b_un.var()}")
        print(f"butter_a_un mean: {butter_a_un.mean()}, butter_b_un mean: {butter_b_un.mean()}, butter_a_un.var: {butter_a_un.var()}, butter_b_un.var: {butter_b_un.var()}")
        if is_theia:
            print(f"a_theia mean: {a_theia.mean()}, b_theia mean: {b_theia.mean()}, a_theia.var: {a_theia.var()}, b_theia.var: {b_theia.var()}")

        factor = round(data.config["measurement_error_factor"], 1)

        a_name, b_name = segment_name_to_a_b_name(name)

        plt.cla()
        plt.plot(ts, a, label="Kalman Filtered", color="steelblue", alpha=0.5, marker=".", markevery=50, linewidth=1)
        plt.plot(ts, a_un, label="Unfiltered Data", color="olive", alpha=0.5, marker=".", markevery=50, linewidth=1)
        if is_theia:
            plt.plot(ts, a_theia, label="Theia", color="darkorange", alpha=0.5, marker=".", markevery=50, linewidth=1)
        # plt.plot(ts, butter_a_un, label="Butterworth Filtered", color="darkorange", alpha=0.5, marker=".", markevery=50)
        plt.xlabel("Time [s]")
        plt.ylabel("Distance [m]")
        plt.legend()
        plt.title(f"Segment Length Distance over Time for {a_name}")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/joint_segment_lengths/factor_{factor}_{segment_name}_upper_segment_{ex_name}.pdf", bbox_inches="tight")
        plt.cla()

        plt.cla()
        plt.plot(ts, b, label="Kalman Filtered", color="steelblue", alpha=0.5, marker=".", markevery=50, linewidth=1)
        plt.plot(ts, b_un, label="Unfiltered Data", color="olive", alpha=0.5, marker=".", markevery=50, linewidth=1)
        if is_theia:
            plt.plot(ts, b_theia, label="Theia", color="darkorange", alpha=0.5, marker=".", markevery=50, linewidth=1)
        # plt.plot(ts, butter_b_un, label="Butterworth Filtered", color="darkorange", alpha=0.5, marker=".", markevery=50)
        plt.xlabel("Time [s]")
        plt.ylabel("Distance [m]")
        plt.legend()
        plt.title(f"Segment Length Distance over Time for {b_name}")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/joint_segment_lengths/factor_{factor}_{segment_name}_lower_segment_{ex_name}.pdf", bbox_inches="tight")
        plt.cla()

        record = (
            ex_name,
            FILTER_NAME,
            factor,
            a_name,
            a.mean(),
            a.var(),
            a_un.mean(),
            a_un.var(),
            a_theia.mean() if is_theia else None,
            a_theia.var() if is_theia else None,
        )
        records.append(record)

        record = (
            ex_name,
            FILTER_NAME,
            factor,
            b_name,
            b.mean(),
            b.var(),
            b_un.mean(),
            b_un.var(),
            b_theia.mean() if is_theia else None,
            b_theia.var() if is_theia else None,
        )
        records.append(record)
    return records



def find_factor_path(factor: float, path: Path) -> Path:
    directories = [element for element in path.iterdir() if element.is_dir()]
    for directory in directories:
        with (directory / "config.json").open(mode="r", encoding="UTF-8") as _f:
            if round(factor, 1) == round(json.load(_f)["measurement_error_factor"], 1):
                return directory
    raise ValueError(f"Factor not found: {factor}")

def compare_qtm_joints_kinect_joints_vel(data: Data, cutoff: float) -> tuple[float, float, float, float, float, float, float, float]:
    kinect_joints = [int(element) for element in [Joint.SHOULDER_LEFT, Joint.ELBOW_LEFT, Joint.WRIST_LEFT]]

    corr = 0
    corr_un = 0

    rmse = 0
    rmse_un = 0

    dtw_dist = 0
    dtw_dist_un = 0

    fr_dist = 0
    fr_dist_un = 0

    for kinect_joint, qtm_joint in zip(kinect_joints, [0, 1, 2]):

        qtm_joints = double_butter(data.qtm_joints[:, qtm_joint, :], sample_frequency=150)
        butter_qtm_velocities = np.zeros_like(qtm_joints)
        butter_qtm_velocities[1:-1] = (qtm_joints[2:] - qtm_joints[:-2]) / (2*(1./150.))
        butter_qtm_velocities[0] = (qtm_joints[1] - qtm_joints[0]) / (1./150)
        butter_qtm_velocities[-1] = (qtm_joints[-1] - qtm_joints[-2]) / (1./150)

        down_qtm_vel = downsample(butter_qtm_velocities, data.qtm_ts, 15)

        joints_vel = data.down_kinect_velocities[:, kinect_joint,:]
        joints_un = double_butter(data.down_kinect_unfiltered_joints[:, kinect_joint, :])

        joints_un_vel = np.zeros_like(joints_un)
        joints_un_vel[1:-1] = (joints_un[2:] - joints_un[:-2]) / (2*(1./15.))
        joints_un_vel[0] = (joints_un[1] - joints_un[0]) / (1./15.)
        joints_un_vel[-1] = (joints_un[-1] - joints_un[-2]) / (1./15.)

        length = min(down_qtm_vel.shape[0], joints_vel.shape[0])
        o = max(int(length * cutoff), 1)

        down_qtm_vel = down_qtm_vel[:length][o:-o]
        joints_vel = joints_vel[:length][o:-o]
        joints_un_vel = joints_un_vel[:length][o:-o]

        corr += np.correlate(down_qtm_vel[:, 0], joints_vel[:, 0])[0] + np.correlate(down_qtm_vel[:, 1], joints_vel[:, 1])[0] + np.correlate(down_qtm_vel[:, 2], joints_vel[:, 2])[0]
        corr_un += np.correlate(down_qtm_vel[:, 0], joints_un_vel[:, 0])[0] + np.correlate(down_qtm_vel[:, 1], joints_un_vel[:, 1])[0] + np.correlate(down_qtm_vel[:, 2], joints_vel[:, 2])[0]

        diff = np.linalg.norm(down_qtm_vel - joints_vel, axis=1)
        rmse += np.sqrt(np.mean(np.power(diff, 2)))

        diff_un = np.linalg.norm(down_qtm_vel - joints_un_vel, axis=1)
        rmse_un += np.sqrt(np.mean(np.power(diff_un, 2)))

        dtw_dist += dtw.dtw(down_qtm_vel, joints_vel, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
        dtw_dist_un += dtw.dtw(down_qtm_vel, joints_un_vel, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

        fr_dist += frechet_dist(down_qtm_vel, joints_vel)
        fr_dist_un += frechet_dist(down_qtm_vel, joints_un_vel)

    return corr, corr_un, rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un

def compare_qtm_joints_kinect_joints_vel_inverted_right(data: Data, cutoff: float) -> tuple[float, float, float, float, float, float, float, float]:
    kinect_joints = [int(element) for element in [Joint.SHOULDER_RIGHT, Joint.ELBOW_RIGHT, Joint.WRIST_RIGHT]]

    corr = 0
    corr_un = 0

    rmse = 0
    rmse_un = 0

    dtw_dist = 0
    dtw_dist_un = 0

    fr_dist = 0
    fr_dist_un = 0

    for kinect_joint, qtm_joint in zip(kinect_joints, [0, 1, 2]):

        qtm_joints = double_butter(data.qtm_joints[:, qtm_joint, :], sample_frequency=150)
        butter_qtm_velocities = np.zeros_like(qtm_joints)
        butter_qtm_velocities[1:-1] = (qtm_joints[2:] - qtm_joints[:-2]) / (2*(1./150.))
        butter_qtm_velocities[0] = (qtm_joints[1] - qtm_joints[0]) / (1./150)
        butter_qtm_velocities[-1] = (qtm_joints[-1] - qtm_joints[-2]) / (1./150)

        down_qtm_vel = downsample(butter_qtm_velocities, data.qtm_ts, 15)

        joints_vel = data.down_kinect_velocities[:, kinect_joint,:]
        joints_vel[:, 0] *= -1
        # joints_un = double_butter(data.down_kinect_unfiltered_joints[:, kinect_joint, :])

        joints_un = data.down_kinect_unfiltered_joints[:, kinect_joint, :]
        joints_un[:, 0] -= 2*data.down_kinect_com[:, 0]
        joints_un = double_butter(joints_un)

        joints_un_vel = np.zeros_like(joints_un)
        joints_un_vel[1:-1] = (joints_un[2:] - joints_un[:-2]) / (2*(1./15.))
        joints_un_vel[0] = (joints_un[1] - joints_un[0]) / (1./15.)
        joints_un_vel[-1] = (joints_un[-1] - joints_un[-2]) / (1./15.)

        length = min(down_qtm_vel.shape[0], joints_vel.shape[0])
        o = max(int(length * cutoff), 1)

        down_qtm_vel = down_qtm_vel[:length][o:-o]
        joints_vel = joints_vel[:length][o:-o]
        joints_un_vel = joints_un_vel[:length][o:-o]

        corr += np.correlate(down_qtm_vel[:, 0], joints_vel[:, 0])[0] + np.correlate(down_qtm_vel[:, 1], joints_vel[:, 1])[0] + np.correlate(down_qtm_vel[:, 2], joints_vel[:, 2])[0]
        corr_un += np.correlate(down_qtm_vel[:, 0], joints_un_vel[:, 0])[0] + np.correlate(down_qtm_vel[:, 1], joints_un_vel[:, 1])[0] + np.correlate(down_qtm_vel[:, 2], joints_vel[:, 2])[0]

        diff = np.linalg.norm(down_qtm_vel - joints_vel, axis=1)
        rmse += np.sqrt(np.mean(np.power(diff, 2)))

        diff_un = np.linalg.norm(down_qtm_vel - joints_un_vel, axis=1)
        rmse_un += np.sqrt(np.mean(np.power(diff_un, 2)))

        dtw_dist += dtw.dtw(down_qtm_vel, joints_vel, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
        dtw_dist_un += dtw.dtw(down_qtm_vel, joints_un_vel, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

        fr_dist += frechet_dist(down_qtm_vel, joints_vel)
        fr_dist_un += frechet_dist(down_qtm_vel, joints_un_vel)

    return corr, corr_un, rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un

def compare_qtm_joints_kinect_joints(data: Data, cutoff: float = 0.15) -> tuple[float, float, float, float, float, float, float, float]:
    kinect_joints = [int(element) for element in [Joint.SHOULDER_RIGHT, Joint.ELBOW_RIGHT, Joint.WRIST_RIGHT]]

    corr = 0
    corr_un = 0

    rmse = 0
    rmse_un = 0

    dtw_dist = 0
    dtw_dist_un = 0

    fr_dist = 0
    fr_dist_un = 0

    for kinect_joint, qtm_joint in zip(kinect_joints, [0, 1, 2]):
        down_qtm = downsample(double_butter(data.qtm_joints[:, qtm_joint, :], 900), data.qtm_ts, 15)
        length = min(down_qtm.shape[0], data.down_kinect_joints.shape[0])
        o = max(int(length * cutoff), 1)

        down_qtm = down_qtm[:length][o:-o]

        joints = data.down_kinect_joints[:, kinect_joint,:][:length][o:-o]
        joints_un = double_butter(data.down_kinect_unfiltered_joints[:, kinect_joint, :])[:length][o:-o]

        corr += np.correlate(down_qtm[:, 0], joints[:, 0])[0] + np.correlate(down_qtm[:, 1], joints[:, 1])[0] + np.correlate(down_qtm[:, 2], joints[:, 2])[0]
        corr_un += np.correlate(down_qtm[:, 0], joints_un[:, 0])[0] + np.correlate(down_qtm[:, 1], joints_un[:, 1])[0] + np.correlate(down_qtm[:, 2], joints[:, 2])[0]

        diff = np.linalg.norm(down_qtm - joints, axis=1)
        rmse += np.sqrt(np.mean(np.power(diff, 2)))

        diff_un = np.linalg.norm(down_qtm - joints_un, axis=1)
        rmse_un += np.sqrt(np.mean(np.power(diff_un, 2)))

        dtw_dist += dtw.dtw(down_qtm, joints, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
        dtw_dist_un += dtw.dtw(down_qtm, joints_un, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

        fr_dist += frechet_dist(down_qtm, joints)
        fr_dist_un += frechet_dist(down_qtm, joints_un)

    return corr, corr_un, rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un



def compare_qtm_joints_kinect_joints_inverted_right(data: Data, cutoff: float = 0.15) -> tuple[float, float, float, float, float, float, float, float]:
    kinect_joints = [int(element) for element in [Joint.SHOULDER_RIGHT, Joint.ELBOW_RIGHT, Joint.WRIST_RIGHT]]

    corr = 0
    corr_un = 0

    rmse = 0
    rmse_un = 0

    dtw_dist = 0
    dtw_dist_un = 0

    fr_dist = 0
    fr_dist_un = 0

    for kinect_joint, qtm_joint in zip(kinect_joints, [0, 1, 2]):
        down_qtm = downsample(double_butter(data.qtm_joints[:, qtm_joint, :], 900), data.qtm_ts, 15)
        length = min(down_qtm.shape[0], data.down_kinect_joints.shape[0])
        o = max(int(length * cutoff), 1)

        down_qtm = down_qtm[:length][o:-o]

        # joints = data.down_kinect_joints[:, kinect_joint,:][:length][o:-o]
        # joints_un = double_butter(data.down_kinect_unfiltered_joints[:, kinect_joint, :])[:length][o:-o]

        joints = data.down_kinect_joints[:, kinect_joint,:]
        joints[:, 0] -= 2*data.down_kinect_com[:, 0]
        joints = joints[:length][o:-o]
        joints_un = data.down_kinect_unfiltered_joints[:, kinect_joint, :]
        joints_un[:, 0] -= 2*data.down_kinect_com[:, 0]
        joints_un = double_butter(joints_un)[:length][o:-o]

        corr += np.correlate(down_qtm[:, 0], joints[:, 0])[0] + np.correlate(down_qtm[:, 1], joints[:, 1])[0] + np.correlate(down_qtm[:, 2], joints[:, 2])[0]
        corr_un += np.correlate(down_qtm[:, 0], joints_un[:, 0])[0] + np.correlate(down_qtm[:, 1], joints_un[:, 1])[0] + np.correlate(down_qtm[:, 2], joints[:, 2])[0]

        diff = np.linalg.norm(down_qtm - joints, axis=1)
        rmse += np.sqrt(np.mean(np.power(diff, 2)))

        diff_un = np.linalg.norm(down_qtm - joints_un, axis=1)
        rmse_un += np.sqrt(np.mean(np.power(diff_un, 2)))

        dtw_dist += dtw.dtw(down_qtm, joints, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
        dtw_dist_un += dtw.dtw(down_qtm, joints_un, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

        fr_dist += frechet_dist(down_qtm, joints)
        fr_dist_un += frechet_dist(down_qtm, joints_un)

    return corr, corr_un, rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un



def compare_qtm_cop_kinect_cop_vel(data: Data, cutoff: float = 0.15) -> tuple[float, float, float, float, float, float, float, float]:
    a = double_butter(data.qtm_cop[:, :2], sample_frequency=900, cutoff=6)
    qtm_vel = np.zeros_like(a)
    qtm_vel[1:-1] = (a[2:] - a[:-2]) / (2*(1./900.))
    qtm_vel[0] = (a[1] - a[0]) / (1./900)
    qtm_vel[-1] = (a[-1] - a[-2]) / (1./900)

    qtm_vel = downsample(qtm_vel, data.qtm_cop_ts, 15)

    kalman_vel = data.down_kinect_com_velocities[:, :2]

    raw = double_butter(data.down_kinect_unfiltered_com)[:, :2]
    raw_vel = np.zeros_like(raw)
    raw_vel[1:-1] = (raw[2:] - raw[:-2]) / (2*(1./15.))
    raw_vel[0] = (raw[1] - raw[0]) / (1./15.)
    raw_vel[-1] = (raw[-1] - raw[-2]) / (1./15.)

    length = min(data.qtm_cop.shape[0], data.down_kinect_com.shape[0])

    qtm_vel = qtm_vel[:length]
    kalman_vel = kalman_vel[:length]
    raw_vel = raw_vel[:length]

    corr = np.correlate(qtm_vel[:, 0], kalman_vel[:, 0])[0] + np.correlate(qtm_vel[:, 1], kalman_vel[:, 1])[0]
    corr_un = np.correlate(qtm_vel[:, 0], raw_vel[:, 0])[0] + np.correlate(qtm_vel[:, 1], raw_vel[:, 1])[0]

    diff = np.linalg.norm(qtm_vel - kalman_vel, axis=1)
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    diff_un = np.linalg.norm(qtm_vel - raw_vel, axis=1)
    rmse_un = np.sqrt(np.mean(np.power(diff_un, 2)))

    dtw_dist = dtw.dtw(qtm_vel, kalman_vel, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
    dtw_dist_un = dtw.dtw(qtm_vel, raw_vel, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

    fr_dist = frechet_dist(qtm_vel, kalman_vel)
    fr_dist_un = frechet_dist(qtm_vel, raw_vel)

    return corr, corr_un, rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un




def compare_qtm_cop_kinect_cop(data: Data, cutoff: float = 0.15) -> tuple[float, float, float, float, float, float, float, float]:
    down_qtm = downsample(double_butter(data.qtm_cop[:, :2], 900), data.qtm_cop_ts, 15)
    length = min(down_qtm.shape[0], data.down_kinect_com.shape[0])
    o = max(int(length * cutoff), 1)

    down_qtm = down_qtm[:length][o:-o]
    com = data.down_kinect_com[:, :2][:length][o:-o]
    com_un = double_butter(data.down_kinect_unfiltered_com[:, :2][:length])[o:-o]

    corr = np.correlate(down_qtm[:, 0], com[:, 0])[0] + np.correlate(down_qtm[:, 1], com[:, 1])[0]
    corr_un = np.correlate(down_qtm[:, 0], com_un[:, 0])[0] + np.correlate(down_qtm[:, 1], com_un[:, 1])[0]

    diff = np.linalg.norm(down_qtm - com, axis=1)
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    diff_un = np.linalg.norm(down_qtm - com_un, axis=1)
    rmse_un = np.sqrt(np.mean(np.power(diff_un, 2)))

    dtw_dist = dtw.dtw(down_qtm, com, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
    dtw_dist_un = dtw.dtw(down_qtm, com_un, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

    fr_dist = frechet_dist(down_qtm, com)
    fr_dist_un = frechet_dist(down_qtm, com_un)

    return corr, corr_un, rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un


def find_best_measurement_error_factor_rmse(experiment_folder: Path, ex_name: str, cutoff: float) -> tuple[Path, float, float, float, float, DetermineFactorRecord, list[tuple]]:
    directories = [element for element in experiment_folder.iterdir() if element.is_dir()]
    RMSEs = []
    dtw_distances = []
    fr_distances = []
    pccs = []
    factors = []

    pltrecords = []
    for directory in tqdm(directories):
        data = cond_load_data(directory)

        #if float(data.config["measurement_error_factor"]) > 1.5:
        #    continue

        rmse = 0
        dtw_distance = 0
        fr_distance = 0
        pcc = 0

        length = min(data.down_kinect_joints.shape[0], data.down_kinect_unfiltered_joints.shape[0])
        o = max(int(length * cutoff), 1)
        factor = float(data.config['measurement_error_factor'])
        if factor == 0:
            #print(np.max(np.abs(data.down_kinect_joints - data.down_kinect_unfiltered_joints)))
            #assert_allclose(data.down_kinect_joints, data.down_kinect_unfiltered_joints, rtol=1e-1)
            continue

        # if factor > 1:
        #     continue

        # if "constraint" in xperiment_type:
        joints = [int(element) for element in [Joint.SHOULDER_RIGHT, Joint.ELBOW_RIGHT, Joint.WRIST_RIGHT]]
        for joint in joints:
            a = data.down_kinect_joints[:length, joint, :][o:-o]
            # b = double_butter(data.down_kinect_unfiltered_joints[:length, joint, :], cutoff=6, once=factor != 0)[o:-o]
            b = double_butter(data.down_kinect_unfiltered_joints[:length, joint, :], cutoff=6)[o:-o]

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

            # diff = np.linalg.norm(b[o:-o] - a[o:-o], axis=1)
            # diff = b - a
            diff = np.linalg.norm(b - a, axis=1)
            error = np.sqrt(np.mean(np.power(diff, 2)))
            rmse += error

            pcc += (np.corrcoef(a[:, 0], b[:, 0])[0, 1] + np.corrcoef(a[:, 1], b[:, 1])[0, 1] + np.corrcoef(a[:, 2], b[:, 2])[0, 1]) / 3

            result = dtw.dtw(a, b, keep_internals=True, step_pattern=dtw.rabinerJuangStepPattern(6, "c"))
            dtw_distance += result.distance

            p = np.column_stack((data.down_kinect_ts[:length][o:-o], a))
            q = np.column_stack((data.down_kinect_ts[:length][o:-o], b))
            fr_distance += frechet_dist(p, q)


        '''
        if False:
            a = data.down_kinect_com[:, :2]
            b = double_butter(data.down_kinect_unfiltered_com[:, :2], N=2)

            # Only take rmse in account for some% of the signal, prevent
            # weighting butterworth problems in the beginning and the end
            diff = np.linalg.norm(b[o:-o] - a[o:-o], axis=1)
            result = np.sqrt(np.mean(np.power(diff, 2)))

            rmse += result

        else:
            raise NotImplementedError(f"Invalid experiment_type: {experiment_type}")
        '''

        # Normalize
        rmse /= 3
        dtw_distance /= 3
        fr_distance /= 3
        pcc /= 3

        factor = data.config["measurement_error_factor"]
        pltrecord = (
            ex_name,
            short_name(FILTER_NAME),
            "Up Right",
            factor,
            rmse,
            dtw_distance,
            fr_distance,
            pcc,
        )
        pltrecords.append(pltrecord)

        RMSEs.append(rmse)
        dtw_distances.append(dtw_distance)
        fr_distances.append(fr_distance)
        factors.append(factor)
        pccs.append(pcc)

    facts = np.array(factors)
    rmses = np.array(RMSEs)
    dtw_dists = np.array(dtw_distances)
    fr_dists = np.array(fr_distances)
    rmse_argmin = np.argmin(rmses)
    dtw_argmin = np.argmin(dtw_dists)
    fr_argmin = np.argmin(fr_dists)

    pccs = np.array(pccs)
    pcc_argmax = np.argmax(pccs)
    idx = np.argsort(facts)

    plt.cla()
    plt.plot(facts[idx][:], rmses[idx][:], label="RMSE", color="steelblue", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
    plt.plot(facts[rmse_argmin], rmses[rmse_argmin], marker="X", ls="None", label=f"Argmin RMSE: {facts[rmse_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title(rf"Ex: {map_ex_name(ex_name)} - RMSE per $\lambda$")
    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_rmse_joints_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.cla()

    plt.plot(facts[idx][:], pccs[idx][:], label="PCC", color="darkorange", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
    plt.plot(facts[pcc_argmax], pccs[pcc_argmax], marker="X", ls="None", label=f"Argmax PCC: {facts[pcc_argmax]:.2f}", color="crimson", alpha=0.6, markersize=8)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("PCC")
    plt.legend()

    plt.title(rf"Ex: {map_ex_name(ex_name)} - PCC per $\lambda$")

    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_pcc_joints_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.cla()

    plt.cla()
    plt.plot(facts[idx][:], dtw_dists[idx][:], label="DTW", color="darkorange", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
    plt.plot(facts[dtw_argmin], dtw_dists[dtw_argmin], marker="X", ls="None", label=f"Argmin DTW: {facts[dtw_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("DTW")
    plt.legend()
    plt.title(rf"Ex: {map_ex_name(ex_name)} - DTW per $\lambda$")
    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_dtw_distance_joints_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.cla()

    plt.cla()
    plt.plot(facts[idx][:], fr_dists[idx][:], label="DFD", color="olive", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
    plt.plot(facts[fr_argmin], fr_dists[fr_argmin], marker="X", ls="None", label=f"Argmin DFD: {facts[fr_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("DFD")
    plt.legend()
    plt.title(rf"Ex: {map_ex_name(ex_name)} - DFD per $\lambda$")
    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_frechet_distance_joints_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.cla()

    record = DetermineFactorRecord(
        ex_name,
        factors[rmse_argmin],
        factors[dtw_argmin],
        factors[fr_argmin],
        factors[pcc_argmax],
    )
    return directories[rmse_argmin], factors[rmse_argmin], factors[dtw_argmin], factors[fr_argmin], factors[pcc_argmax], record, pltrecords

def find_best_measurement_error_factor_rmse_on_velocity(experiment_folder: Path, ex_name: str, cutoff: float) -> tuple[Path, float, float, float, float, DetermineFactorRecord, list[tuple]]:
    """Find the best measurement error factor through comparing velocities."""
    directories = [element for element in experiment_folder.iterdir() if element.is_dir()]
    RMSEs = []
    dtw_distances = []
    fr_distances = []
    pccs = []
    factors = []

    pltrecords = []
    for directory in tqdm(directories):
        data = cond_load_data(directory)

        length = min(data.down_kinect_joints.shape[0], data.down_kinect_unfiltered_joints.shape[0])
        o = int(length * cutoff)
        factor = float(data.config['measurement_error_factor'])

        if factor == 0:
            continue

        # if factor > 1:
        #     continue

        rmse = 0
        dtw_distance = 0
        fr_distance = 0
        corr_offset = 0
        pcc = 0
        for idx, joint in enumerate([Joint.SHOULDER_RIGHT, Joint.ELBOW_RIGHT, Joint.WRIST_RIGHT]):
            a_vel = data.down_kinect_velocities[:length, int(joint), :]
            # If we have a kalman filter factor then we also want to introduce butterworth filter lag
            b = double_butter(data.down_kinect_unfiltered_joints[:length, int(joint), :], cutoff=6, N=2, once=False)
            # c = data.down_kinect_unfiltered_joints[:length, int(joint), :]

            b_vel = np.zeros_like(b)
            b_vel[1:-1] = (b[2:] - b[:-2]) / (2*(1./15.))
            b_vel[0] = (b[1] - b[0]) / (1./15.)
            b_vel[-1] = (b[-1] - b[-2]) / (1./15.)

            # b_vel = double_butter(b_vel, sample_frequency=15, cutoff=6)

            '''
            qtm_joints = double_butter(data.qtm_joints[:, idx, :], 150)

            butter_qtm_velocities = np.zeros_like(qtm_joints)
            butter_qtm_velocities[1:-1] = (qtm_joints[2:] - qtm_joints[:-2]) / (2*(1./150.))
            butter_qtm_velocities[0] = (qtm_joints[1] - qtm_joints[0]) / (1./150)
            butter_qtm_velocities[-1] = (qtm_joints[-1] - qtm_joints[-2]) / (1./150)
            d_vel = downsample(butter_qtm_velocities, data.qtm_ts, 15)
            '''

            # extract finite velocity
            # downsampled have 15 hz frequency

            '''
            if joint == Joint.WRIST_LEFT and once and i == 2 and 0 <= factor < 1.05:
                once = False

                c_vel = np.zeros_like(c)
                c_vel[1:-1] = (c[2:] - c[:-2]) / (2*(1./15.))
                c_vel[0] = (c[1] - c[0]) / (1./15.)
                c_vel[-1] = (c[-1] - c[-2]) / (1./15.)

                off = np.argmax(signal.correlate(b_vel[o:-o], a_vel[o:-o])) - (len(a_vel) - 2*o)
                time = off * (1/15)
                plt.plot(data.down_kinect_ts + time, a_vel, label="kalman vel")
                plt.plot(data.down_kinect_ts, b_vel, label="butter fd vel")
                # plt.plot(data.down_kinect_ts, c_vel, label="raw fd vel")
                qtm_ts = np.arange(0, d_vel.shape[0]) * (1./15.)
                plt.plot(qtm_ts, d_vel, label="qtm fd vel")
                plt.legend()
                plt.title(f"Error factor: {data.config['measurement_error_factor']}");
                plt.show();
                plt.cla();
                result = dtw.dtw(a_vel, b_vel, keep_internals=True, step_pattern=dtw.rabinerJuangStepPattern(6, "c"))

                result.plot(type="twoway",offset=-2)
                plt.show()
                plt.cla()
            '''

            '''
            # Adjust for lag by shifting based on cross correlation
            lag = np.argmax(signal.correlate(b_vel[o:-o], a_vel[o:-o])) - (len(a_vel) - 2 * o)

            # Factor 0 means a real high velocity error, correlation
            # doesn't realy provide good results
            if factor == 0:
                lag = 0

            if o < np.abs(lag):
                # print("Joint {joint} axis {i}")
                # print("Lag is to big, reseting it to 0")
                lag = 0
            '''

            # disabel lag
            lag = 0

            diff = np.linalg.norm(b_vel[o:-o] - a_vel[o:-o], axis=1)
            error = np.sqrt(np.mean(np.power(diff, 2)))
            rmse += error

            result = dtw.dtw(a_vel, b_vel, keep_internals=True, step_pattern=dtw.rabinerJuangStepPattern(6, "c"))
            dtw_distance += result.distance

            pcc += (np.corrcoef(a_vel[:, 0], b_vel[:, 0])[0, 1] + np.corrcoef(a_vel[:, 1], b_vel[:, 1])[0, 1] + np.corrcoef(a_vel[:, 2], b_vel[:, 2])[0, 1]) / 3

            # p = np.column_stack((np.arange(0, len(a_vel))[:20], a_vel[:20]))
            # q = np.column_stack((np.arange(0, len(b_vel))[:20], b_vel[:20]))
            # fr_distance += frdist(p, q)
            p = np.column_stack((data.down_kinect_ts[:length], a_vel))
            q = np.column_stack((data.down_kinect_ts[:length], b_vel))
            fr_distance += frechet_dist(p, q)

        # Normalize
        rmse /= 3
        dtw_distance /= 3
        fr_distance /= 3
        pcc /= 3

        factor = data.config["measurement_error_factor"]
        pltrecord = (
            ex_name,
            FILTER_NAME,
            "Up Right",
            factor,
            rmse,
            dtw_distance,
            fr_distance,
            pcc,
        )
        pltrecords.append(pltrecord)

        RMSEs.append(rmse)
        dtw_distances.append(dtw_distance)
        fr_distances.append(fr_distance)
        factors.append(factor)
        pccs.append(pcc)

    facts = np.array(factors)
    rmses = np.array(RMSEs)
    dtw_dists = np.array(dtw_distances)
    fr_dists = np.array(fr_distances)
    rmse_argmin = np.argmin(rmses)
    dtw_argmin = np.argmin(dtw_dists)
    fr_argmin = np.argmin(fr_dists)

    pccs = np.array(pccs)
    pcc_argmax = np.argmax(pccs)

    idx = np.argsort(facts)

    plt.cla()
    plt.plot(facts[idx][:], rmses[idx][:], label="RMSE", color="steelblue", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
    plt.plot(facts[rmse_argmin], rmses[rmse_argmin], marker="X", ls="None", label=f"Argmin RMSE: {facts[rmse_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title(rf"Ex: {map_ex_name(ex_name)} - RMSE per $\lambda$")
    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_rmse_velocity_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.cla()

    '''
    plt.plot(facts[idx][:], corrs[idx][:], marker="X", ls="None", label="Correlation")
    plt.xlabel("$\lambda$")
    plt.ylabel("Correlation offset")
    plt.legend()

    jump_idx = np.argmax(corrs[idx][:] != 0)
    plt.title(f"{facts[idx][:][jump_idx]}:{factors[rmse_argmin]}- Ex: {map_ex_name(ex_name)} Correlation offset : measurement error factor")
    print(f"Correlation jump {facts[idx][:][jump_idx]}:Factor argmin {factors[rmse_argmin]}")

    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_rmse_velocity_correlation_offset_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.cla()
    '''

    plt.cla()
    plt.plot(facts[idx][:], dtw_dists[idx][:], label="DTW", color="darkorange", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
    plt.plot(facts[dtw_argmin], dtw_dists[dtw_argmin], marker="X", ls="None", label=f"Argmin DTW: {facts[dtw_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("DTW")
    plt.legend()
    plt.title(rf"Ex: {map_ex_name(ex_name)} - DTW per $\lambda$")
    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_dtw_distance_velocity_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.cla()

    plt.plot(facts[idx][:], pccs[idx][:], label="PCC", color="darkorange", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
    plt.plot(facts[pcc_argmax], pccs[pcc_argmax], marker="X", ls="None", label=f"Argmax PCC: {facts[pcc_argmax]:.2f}", color="crimson", alpha=0.6, markersize=8)
    plt.xlabel("$\lambda$")
    plt.ylabel("PCC")
    plt.legend()

    plt.title(rf"Ex: {map_ex_name(ex_name)} - PCC per $\lambda$")

    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_pcc_velocity_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.cla()

    plt.cla()
    plt.plot(facts[idx][:], fr_dists[idx][:], label="DFD", color="olive", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
    plt.plot(facts[fr_argmin], fr_dists[fr_argmin], marker="X", ls="None", label=f"Argmin DFD: {facts[fr_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("DFD")
    plt.legend()
    plt.title(rf"Ex: {map_ex_name(ex_name)} - DFD per $\lambda$")
    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_frechet_distance_velocity_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.cla()

    record = DetermineFactorRecord(
        ex_name,
        factors[rmse_argmin],
        factors[dtw_argmin],
        factors[fr_argmin],
        factors[pcc_argmax],
    )
    return directories[rmse_argmin], factors[rmse_argmin], factors[dtw_argmin], factors[fr_argmin], factors[pcc_argmax], record, pltrecords

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
    ex_name = os.path.basename(experiment_folder)
    plt.plot(np.array(factors), corrs, marker="X", ls="None", label="Correlation")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Correlation")
    plt.legend()
    plt.title(f"Ex: {map_ex_name(ex_name)} Correlation per measurement error factor")
    # plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_corr_{experiment_type}_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
    plt.show()
    plt.cla()
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
    # assert len(corrs) == len(directories)

    plt.plot(np.array(factors), corrs, marker="X", ls="None", label="Correlation")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Correlation")
    plt.legend()
    plt.title(f"Ex: {map_ex_name(ex_name)} Correlation per measurement error factor")
    plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor/factors_corr_{os.path.basename(experiment_folder)}.pdf", bbox_inches="tight")
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
    if SHOW:
        plt.show()
    plt.cla()


def determine_minimum_against_ground_truth_theia(experiment_folder: Path, ex_name: str, cutoff: float = 0.1) -> tuple[DetermineFactorRecord, DetermineFactorRecord, list[tuple], list[tuple]]:
    directories = [element for element in experiment_folder.iterdir() if element.is_dir()]
    os.makedirs(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/{ex_name}", exist_ok=True)
    pltrecords = []
    vel_pltrecords = []
    over_ride = [([Joint.SHOULDER_RIGHT, Joint.ELBOW_RIGHT, Joint.WRIST_RIGHT], [TheiaJoint.SHOULDER_RIGHT, TheiaJoint.ELBOW_RIGHT, TheiaJoint.WRIST_RIGHT], "UP_RIGHT")]
    # for (kinect_joints, theia_joints, segment_name) in JOINT_SEGMENTS:
    for (kinect_joints, theia_joints, segment_name) in over_ride:
        corr_a = []
        frechet_a = []
        dtw_a = []
        rmse_a = []
        factors = []

        vel_corr_a = []
        vel_frechet_a = []
        vel_dtw_a = []
        vel_rmse_a = []
        vel_factors = []


        for directory in tqdm(directories):
            data: TheiaData = cond_load_data(directory)
            if data.config["measurement_error_factor"] == 0:
                continue

            theia_ts = np.arange(data.theia_tensor.shape[0]) * (1./120.)

            rmse = 0
            dtwsum = 0
            fr = 0
            corr = 0

            vel_rmse = 0
            vel_dtwsum = 0
            vel_fr = 0
            vel_corr = 0

            for kinect_joint, theia_joint in zip(kinect_joints, theia_joints):
                d_t = downsample(double_butter(data.theia_tensor[:, int(theia_joint), :], 120), theia_ts, 15)
                d_f = downsample(data.kinect_joints[:, int(kinect_joint), :], data.kinect_ts, 15)

                vel_d_t = downsample(double_butter(central_diff(data.theia_tensor[:, int(theia_joint), :], 120), 120), theia_ts, 15)
                vel_d_f = downsample(data.kinect_velocities[:, int(kinect_joint), :], data.kinect_ts, 15)

                length = min(d_t.shape[0], d_f.shape[0])
                o = int(length * cutoff)

                d_t = d_t[:length][o:-o]
                d_f = d_f[:length][o:-o]

                vel_d_t = vel_d_t[:length][o:-o]
                vel_d_f = vel_d_f[:length][o:-o]

                corr += (np.corrcoef(d_t[:, 0], d_f[:, 0])[0, 1] + np.corrcoef(d_t[:, 1], d_f[:, 1])[0, 1] + np.corrcoef(d_t[:, 2], d_f[:, 2])[0, 1]) / 3
                vel_corr += (np.corrcoef(vel_d_t[:, 0], vel_d_f[:, 0])[0, 1] + np.corrcoef(vel_d_t[:, 1], vel_d_f[:, 1])[0, 1] + np.corrcoef(vel_d_t[:, 2], vel_d_f[:, 2])[0, 1]) / 3

                diff = np.linalg.norm(d_t - d_f, axis=1)
                rmse += np.sqrt(np.mean(np.power(diff, 2)))

                vel_diff = np.linalg.norm(vel_d_t - vel_d_f, axis=1)
                vel_rmse += np.sqrt(np.mean(np.power(vel_diff, 2)))

                ts = np.arange(len(d_t))
                d_t = np.column_stack((ts, d_t))
                d_f = np.column_stack((ts, d_f))

                vel_d_t = np.column_stack((ts, vel_d_t))
                vel_d_f = np.column_stack((ts, vel_d_f))

                fr += frechet_dist(d_t, d_f)
                vel_fr += frechet_dist(vel_d_t, vel_d_f)

                dtwsum += dtw.dtw(d_t, d_f, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
                vel_dtwsum += dtw.dtw(vel_d_t, vel_d_f, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

            # Normalize data
            corr /= 3
            fr /= 3
            dtwsum /= 3
            rmse /= 3

            vel_corr /= 3
            vel_fr /= 3
            vel_dtwsum /= 3
            vel_rmse /= 3

            factor = data.config["measurement_error_factor"]
            pltrecord = (
                ex_name,
                FILTER_NAME,
                segment_name,
                factor,
                rmse,
                dtwsum,
                fr,
                corr,
            )
            pltrecords.append(pltrecord)

            vel_pltrecord = (
                ex_name,
                FILTER_NAME,
                segment_name,
                factor,
                vel_rmse,
                vel_dtwsum,
                vel_fr,
                vel_corr,
            )
            vel_pltrecords.append(vel_pltrecord)

            corr_a.append(corr)
            frechet_a.append(fr)
            dtw_a.append(dtwsum)
            rmse_a.append(rmse)

            vel_corr_a.append(vel_corr)
            vel_frechet_a.append(vel_fr)
            vel_dtw_a.append(vel_dtwsum)
            vel_rmse_a.append(vel_rmse)

            factors.append(factor)

        facts = np.array(factors)
        rmses = np.array(rmse_a)
        dtw_dists = np.array(dtw_a)
        fr_dists = np.array(frechet_a)
        corrs = np.array(corr_a)

        rmse_argmin = np.argmin(rmses)
        dtw_argmin = np.argmin(dtw_dists)
        fr_argmin = np.argmin(fr_dists)
        pcc_argmax = np.argmax(corrs)

        idx = np.argsort(facts)

        vel_rmses = np.array(vel_rmse_a)
        vel_dtw_dists = np.array(vel_dtw_a)
        vel_fr_dists = np.array(vel_frechet_a)
        vel_corrs = np.array(vel_corr_a)

        vel_rmse_argmin = np.argmin(vel_rmses)
        vel_dtw_argmin = np.argmin(vel_dtw_dists)
        vel_fr_argmin = np.argmin(vel_fr_dists)
        vel_corr_argmax = np.argmax(vel_corrs)

        vel_idx = np.argsort(facts)

        plt.cla()
        plt.plot(facts[idx][:], rmses[idx][:], label="RMSE", color="steelblue", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
        plt.plot(facts[rmse_argmin], rmses[rmse_argmin], marker="X", ls="None", label=f"Argmin RMSE: {facts[rmse_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
        plt.xlabel(r"$\lambda$")
        plt.ylabel("RMSE")
        plt.legend()
        plt.title(rf"Ex: {map_ex_name(ex_name)} - RMSE per $\lambda$")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/{ex_name}/rmse_{segment_name}.pdf", bbox_inches="tight")
        plt.cla()

        plt.cla()
        plt.plot(facts[idx][:], dtw_dists[idx][:], label="DTW", color="darkorange", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
        plt.plot(facts[dtw_argmin], dtw_dists[dtw_argmin], marker="X", ls="None", label=f"Argmin DTW: {facts[dtw_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
        plt.xlabel(r"$\lambda$")
        plt.ylabel("DTW")
        plt.legend()
        plt.title(rf"Ex: {map_ex_name(ex_name)} - DTW per $\lambda$")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/{ex_name}/dtw_{segment_name}.pdf", bbox_inches="tight")
        plt.cla()

        plt.cla()
        plt.plot(facts[idx][:], fr_dists[idx][:], label="DFD", color="olive", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
        plt.plot(facts[fr_argmin], fr_dists[fr_argmin], marker="X", ls="None", label=f"Argmin DFD: {facts[fr_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
        plt.xlabel(r"$\lambda$")
        plt.ylabel("DFD")
        plt.legend()
        plt.title(rf"Ex: {map_ex_name(ex_name)} - DFD per $\lambda$")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/{ex_name}/frechet_{segment_name}.pdf", bbox_inches="tight")
        plt.cla()

        plt.cla()
        plt.plot(facts[idx][:], corrs[idx][:], label="PCC", color="olive", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
        plt.plot(facts[pcc_argmax], corrs[pcc_argmax], marker="X", ls="None", label=f"Argmax PCC: {facts[pcc_argmax]:.2f}", color="crimson", alpha=0.6, markersize=8)
        plt.xlabel(r"$\lambda$")
        plt.ylabel("PCC")
        plt.legend()
        plt.title(rf"Ex: {map_ex_name(ex_name)} - PCC per $\lambda$")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/{ex_name}/corr_{segment_name}.pdf", bbox_inches="tight")
        plt.cla()


        print(f"Segment: {segment_name}")
        print(factors[rmse_argmin], factors[dtw_argmin], factors[fr_argmin], factors[pcc_argmax])
        record = DetermineFactorRecord(
            ex_name,
            factors[rmse_argmin],
            factors[dtw_argmin],
            factors[fr_argmin],
            factors[pcc_argmax],
        )

        plt.cla()
        plt.plot(facts[vel_idx][:], vel_rmses[vel_idx][:], label="RMSE", color="steelblue", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
        plt.plot(facts[vel_rmse_argmin], vel_rmses[vel_rmse_argmin], marker="X", ls="None", label=f"Argmin RMSE: {facts[vel_rmse_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
        plt.xlabel(r"$\lambda$")
        plt.ylabel("RMSE")
        plt.legend()
        plt.title(rf"Ex: {map_ex_name(ex_name)} - RMSE per $\lambda$")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/{ex_name}/vel_rmse_{segment_name}.pdf", bbox_inches="tight")
        plt.cla()

        plt.cla()
        plt.plot(facts[vel_idx][:], vel_dtw_dists[vel_idx][:], label="DTW", color="darkorange", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
        plt.plot(facts[vel_dtw_argmin], vel_dtw_dists[vel_dtw_argmin], marker="X", ls="None", label=f"Argmin DTW: {facts[vel_dtw_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
        plt.xlabel(r"$\lambda$")
        plt.ylabel("DTW")
        plt.legend()
        plt.title(rf"Ex: {map_ex_name(ex_name)} - DTW per $\lambda$")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/{ex_name}/vel_dtw_{segment_name}.pdf", bbox_inches="tight")
        plt.cla()

        plt.cla()
        plt.plot(facts[vel_idx][:], vel_fr_dists[vel_idx][:], label="DFD", color="olive", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
        plt.plot(facts[vel_fr_argmin], vel_fr_dists[vel_fr_argmin], marker="X", ls="None", label=f"Argmin DFD: {facts[vel_fr_argmin]:.2f}", color="crimson", alpha=0.6, markersize=8)
        plt.xlabel(r"$\lambda$")
        plt.ylabel("DFD")
        plt.legend()
        plt.title(rf"Ex: {map_ex_name(ex_name)} - DFD per $\lambda$")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/{ex_name}/vel_frechet_{segment_name}.pdf", bbox_inches="tight")
        plt.cla()

        plt.cla()
        plt.plot(facts[vel_idx][:], vel_corrs[vel_idx][:], label="PCC", color="olive", marker='.', markersize=8, markeredgecolor='black', alpha=0.4, linewidth=4)
        plt.plot(facts[vel_corr_argmax], vel_corrs[vel_corr_argmax], marker="X", ls="None", label=f"Argmax PCC: {facts[vel_corr_argmax]:.2f}", color="crimson", alpha=0.6, markersize=8)
        plt.xlabel(r"$\lambda$")
        plt.ylabel("PCC Correlation")
        plt.legend()
        plt.title(rf"Ex: {map_ex_name(ex_name)} - PCC per $\lambda$")
        plt.savefig(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/{ex_name}/vel_corr_{segment_name}.pdf", bbox_inches="tight")
        plt.cla()

        print(f"Segment velocity: {segment_name}")
        print(factors[vel_rmse_argmin], factors[vel_dtw_argmin], factors[vel_fr_argmin], factors[vel_corr_argmax])
        vel_record = DetermineFactorRecord(
            ex_name,
            factors[vel_rmse_argmin],
            factors[vel_dtw_argmin],
            factors[vel_fr_argmin],
            factors[vel_corr_argmax],
        )
        return record, vel_record, pltrecords, vel_pltrecords




def determine_minimum_against_ground_truth(args):
    ranges = np.arange(0, 100, 2.5)
    corr_a = []
    corr_b = []
    corr_c = []

    frechet_a = []
    frechet_b = []
    frechet_c = []

    dtw_a = []
    dtw_b = []
    dtw_c = []

    rmse_a = []
    rmse_b = []
    rmse_c = []

    for i in ranges:
        data1 = load_processed_data(find_factor_path(i, Path(args.experiment_folder)))
        data2 = load_processed_data(find_factor_path(i, Path(args.second_folder)))
        # plt.plot(data1.kinect_ts, data1.kinect_joints[:, int(Joint.WRIST_LEFT), 2], label="1 joints")
        plt.plot(data2.kinect_ts, data2.kinect_joints[:, int(Joint.WRIST_LEFT), 2], label="2 joints")
        plt.plot(data2.kinect_ts, data2.kinect_unfiltered_joints[:, int(Joint.WRIST_LEFT), 2], label="raw")
        #plt.plot(data1.kinect_ts, data1.kinect_predictions[:, int(Joint.WRIST_LEFT), 2], label="1 predict")
        #plt.plot(data2.kinect_ts, data2.kinect_predictions[:, int(Joint.WRIST_LEFT), 2], label="2 predict")
        plt.plot(data1.qtm_ts, data1.qtm_joints[:, 2, 2], label="truth")
        plt.legend()
        plt.cla()

        # plt.show()

        d_q = downsample(double_butter(data1.qtm_joints[:, 2, 2], 150), data1.qtm_ts, 15)
        d_a = downsample(data1.kinect_joints[:, int(Joint.WRIST_LEFT), 2], data1.kinect_ts, 15)
        d_b = downsample(data2.kinect_joints[:, int(Joint.WRIST_LEFT), 2], data2.kinect_ts, 15)
        d_c = downsample(data1.kinect_unfiltered_joints[:, int(Joint.WRIST_LEFT), 2], data1.kinect_ts, 15)
        length = min(d_a.shape[0], d_b.shape[0])

        d_q = d_q[:length]

        d_a = d_a[:length]
        d_b = d_b[:length]
        d_c = d_c[:length]

        corr_a.append(np.corrcoef(d_a, d_q)[0, 1])
        corr_b.append(np.corrcoef(d_b, d_q)[0, 1])
        corr_c.append(np.corrcoef(d_c, d_q)[0, 1])

        ts = np.arange(len(d_a))
        d_q = np.column_stack((ts, d_q))
        d_a = np.column_stack((ts, d_a))
        d_b = np.column_stack((ts, d_b))
        d_c = np.column_stack((ts, d_c))

        frechet_a.append(frechet_dist(d_a, d_q))
        frechet_b.append(frechet_dist(d_b, d_q))
        frechet_c.append(frechet_dist(d_c, d_q))

        dtw_a.append(dtw.dtw(d_a, d_q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance)
        dtw_b.append(dtw.dtw(d_b, d_q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance)
        dtw_c.append(dtw.dtw(d_c, d_q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance)

        rmse_a.append(np.sqrt(np.mean(np.power(d_a[:, 1] - d_q[:, 1], 2), axis=0)))
        rmse_b.append(np.sqrt(np.mean(np.power(d_b[:, 1] - d_q[:, 1], 2), axis=0)))
        rmse_c.append(np.sqrt(np.mean(np.power(d_c[:, 1] - d_q[:, 1], 2), axis=0)))

    corr_mean = np.array(corr_c).mean()
    plt.plot(corr_a / corr_mean, label="filter with acc correlation")
    plt.plot(corr_b / corr_mean, label="filter wo acc correlation")
    plt.plot(corr_a / corr_mean, label="raw correlation")

    frechet_mean = np.array(frechet_c).mean()
    plt.plot(frechet_a / frechet_mean, label="filter with acc frechet")
    plt.plot(frechet_b / frechet_mean, label="filter wo acc frechet")
    plt.plot(frechet_c / frechet_mean, label="raw frechet")

    dtw_mean = np.array(dtw_c).mean()
    plt.plot(dtw_a / dtw_mean, label="filter with acc dtw")
    plt.plot(dtw_b / dtw_mean, label="filter wo acc dtw")
    plt.plot(dtw_c / dtw_mean, label="raw dtw")

    rmse_mean = np.array(rmse_c).mean()
    plt.plot(rmse_a / rmse_mean, label="filter with acc rmse")
    plt.plot(rmse_b / rmse_mean, label="filter wo acc rmse")
    plt.plot(rmse_c / rmse_mean, label="raw rmse")

    plt.legend()
    plt.show()


def all_ex_find_best_measurement_error_factor_rmse(path: Path):
    # Against Butterworth Kinect
    table = GenericTableGeneration(["Filter Name", "RMSE", "DTW", "DFD", "PCC"], r"Mean Optimal $\lambda$ Evaluated on Joint Positions", "mean-optimal-lambda")
    vel_table = GenericTableGeneration(["Filter Name", "RMSE", "DTW", "DFD", "PCC"], r"Mean Optimal $\lambda$ Evaluated on Joint Velocities", "mean-optimal-lambda-vel")
    plot_records = []
    vel_plot_records = []
    for filter_type in FILTER_TYPES:
        global FILTER_NAME
        FILTER_NAME = filter_type
        generator = DetermineFactorGenerateTable(filter_type=filter_type, eval_type="Joint Positions", records=[])
        vel_generator = DetermineFactorGenerateTable(filter_type=filter_type, eval_type="Joint Velocities", records=[], vel=True)
        for ex_name in EX_NAMES:
            ex_path = path / filter_type / ex_name
            _, _, _, _, _, record, pltrecords = find_best_measurement_error_factor_rmse(ex_path, ex_name, 0.1)
            plot_records.extend(pltrecords)
            generator.append(record)
            _, _, _, _, _, vel_record, vel_pltrecords = find_best_measurement_error_factor_rmse_on_velocity(ex_path, ex_name, 0.1)
            vel_plot_records.extend(vel_pltrecords)
            vel_generator.append(vel_record)

        out_path = Path(f"./results/experiments/{filter_type}/determine_factor/table.tex")
        generator.generate_table(out_path)
        mean = generator.mean()
        table.append([
            "\\"+filter_type,
            str(round(mean.rmse, 1)),
            str(round(mean.dtw, 1)),
            str(round(mean.dfd, 1)),
            str(round(mean.pcc, 1)),
        ])

        vel_out_path = Path(f"./results/experiments/{filter_type}/determine_factor/vel_table.tex")
        vel_generator.generate_table(vel_out_path)
        vel_mean = vel_generator.mean()
        vel_table.append([
            "\\"+filter_type,
            str(round(vel_mean.rmse, 1)),
            str(round(vel_mean.dtw, 1)),
            str(round(vel_mean.dfd, 1)),
            str(round(vel_mean.pcc, 1)),
        ])

    for filter_type in FILTER_TYPES:
        mean_out_path = Path(f"./results/experiments/{filter_type}/determine_factor/mean_table.tex")
        mean_vel_out_path = Path(f"./results/experiments/{filter_type}/determine_factor/mean_vel_table.tex")
        table.generate_table(mean_out_path)
        vel_table.generate_table(mean_vel_out_path)


    # Against Truth
    table = GenericTableGeneration(["Filter Name", "RMSE", "DTW", "DFD", "PCC"], r"Mean Optimal $\lambda$ Evaluated on Joint Positions Against Ground Truth", "mean-optimal-lambda-truth")
    vel_table = GenericTableGeneration(["Filter Name", "RMSE", "DTW", "DFD", "PCC"], r"Mean Optimal $\lambda$ Evaluated on Joint Velocities Against Ground Truth", "mean-optimal-lambda-vel-truth")
    truth_plot_records = []
    truth_vel_plot_records = []
    for filter_type in FILTER_TYPES:
        FILTER_NAME = filter_type
        generator = DetermineFactorGenerateTable(filter_type=filter_type, eval_type="Joint Positions Against Ground Truth", records=[])
        vel_generator = DetermineFactorGenerateTable(filter_type=filter_type, eval_type="Joint Velocities Against Ground Truth", records=[], vel=True)
        for ex_name in THEIA_EX_NAMES:
            ex_path = path / filter_type / ex_name
            record, vel_record, pltrecords, vel_pltrecords = determine_minimum_against_ground_truth_theia(ex_path, ex_name, 0.1)
            generator.append(record)
            truth_plot_records.extend(pltrecords)
            truth_vel_plot_records.extend(vel_pltrecords)
            vel_generator.append(vel_record)

        out_path = Path(f"./results/experiments/{filter_type}/determine_factor_against_truth/table.tex")
        generator.generate_table(out_path)
        mean = generator.mean()
        table.append([
            "\\"+filter_type,
            str(round(mean.rmse, 3)),
            str(round(mean.dtw, 3)),
            str(round(mean.dfd, 3)),
            str(round(mean.pcc, 3)),
        ])

        vel_out_path = Path(f"./results/experiments/{filter_type}/determine_factor_against_truth/vel_table.tex")
        vel_generator.generate_table(vel_out_path)
        vel_mean = vel_generator.mean()
        vel_table.append([
            "\\"+filter_type,
            str(round(vel_mean.rmse, 3)),
            str(round(vel_mean.dtw, 3)),
            str(round(vel_mean.dfd, 3)),
            str(round(vel_mean.pcc, 3)),
        ])

    for filter_type in FILTER_TYPES:
        mean_out_path = Path(f"./results/experiments/{filter_type}/determine_factor_against_truth/mean_table.tex")
        mean_vel_out_path = Path(f"./results/experiments/{filter_type}/determine_factor_against_truth/mean_vel_table.tex")
        table.generate_table(mean_out_path)
        vel_table.generate_table(mean_vel_out_path)

    # Plot determine factor mean all filters different metrics
    for metric, ylabel in zip(METRIC_NAMES, YLABELS):
        recorddata = np.array(plot_records, dtype=[
            ("Experiment Name", f"U{len('s30001')}"),
            ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
            ("Segment Name", f"U{len('Upper Arm Right')}"),
            ("Factor", "f"),
            ("RMSE", "f"),
            ("DTW", "f"),
            ("DFD", "f"),
            ("PCC", "f"),
        ])
        dataframe = pd.DataFrame.from_records(recorddata)
        sns.set_style("darkgrid")
        len_joints = len(MATCHING_JOINTS)
        linestyles=(["-"] * 8 + [":"] * 8)[:len_joints] + ["-."]
        ax = sns.catplot(
            data=dataframe,
            x="Factor",
            y=metric,
            kind="point",
            hue="Filter Name",
            linestyles=linestyles,
            markersize=0,
            linewidth=1,
            native_scale=True,
            errorbar=None
        )
        # ax.set_xticklabels(rotation=40, ha="right")
        # ticks = ax.ax.get_xticklabels()
        # plt.setp(ax.ax.get_xticklabels(), visible=False)
        # plt.setp(ticks[::6], visible=True)
        plt.xlabel(r"$\lambda$")
        # Otherwise take default
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(rf"Mean {metric} for different $\lambda$ Evaluated on Joint Position")
        os.makedirs(f"./results/experiments/determine_factor/sx000x", exist_ok=True)
        plt.savefig(f"./results/experiments/determine_factor/sx000x/{metric}_over_factor.pdf", bbox_inches="tight")
        plt.cla()

        recorddata = np.array(vel_plot_records, dtype=[
            ("Experiment Name", f"U{len('s30001')}"),
            ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
            ("Segment Name", f"U{len('Upper Arm Right')}"),
            ("Factor", "f"),
            ("RMSE", "f"),
            ("DTW", "f"),
            ("DFD", "f"),
            ("PCC", "f"),
        ])
        dataframe = pd.DataFrame.from_records(recorddata)
        sns.set_style("darkgrid")
        len_joints = len(MATCHING_JOINTS)
        linestyles=(["-"] * 8 + [":"] * 8)[:len_joints] + ["-."]
        ax = sns.catplot(
            data=dataframe,
            x="Factor",
            y=metric,
            kind="point",
            hue="Filter Name",
            linestyles=linestyles,
            markersize=0,
            linewidth=1,
            native_scale=True,
            errorbar=None
        )
        # ax.set_xticklabels(rotation=40, ha="right")
        # ticks = ax.ax.get_xticklabels()
        # plt.setp(ax.ax.get_xticklabels(), visible=False)
        # plt.setp(ticks[::6], visible=True)
        plt.xlabel(r"$\lambda$")
        # Otherwise take default
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(rf"Mean {metric} for different $\lambda$ Evaluated on Joint Velocities")
        os.makedirs(f"./results/experiments/determine_factor/sx000x", exist_ok=True)
        plt.savefig(f"./results/experiments/determine_factor/sx000x/{metric}_over_factor_vel.pdf", bbox_inches="tight")
        plt.cla()

    # Plot determine factor mean all filters different metrics
    for metric, ylabel in zip(METRIC_NAMES, YLABELS):
        recorddata = np.array(truth_plot_records, dtype=[
            ("Experiment Name", f"U{len('s30001')}"),
            ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
            ("Segment Name", f"U{len('Upper Arm Right')}"),
            ("Factor", "f"),
            ("RMSE", "f"),
            ("DTW", "f"),
            ("DFD", "f"),
            ("PCC", "f"),
        ])
        dataframe = pd.DataFrame.from_records(recorddata)
        sns.set_style("darkgrid")
        len_joints = len(MATCHING_JOINTS)
        linestyles=(["-"] * 8 + [":"] * 8)[:len_joints] + ["-."]
        ax = sns.catplot(
            data=dataframe,
            x="Factor",
            y=metric,
            kind="point",
            hue="Filter Name",
            linestyles=linestyles,
            markersize=0,
            linewidth=1,
            native_scale=True,
            errorbar=None
        )
        # ax.set_xticklabels(rotation=40, ha="right")
        # ticks = ax.ax.get_xticklabels()
        # plt.setp(ax.ax.get_xticklabels(), visible=False)
        # plt.setp(ticks[::6], visible=True)
        plt.xlabel(r"$\lambda$")
        # Otherwise take default
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(rf"Mean {metric} for different $\lambda$ Evaluated on Joint Position Against Ground Truth")
        os.makedirs(f"./results/experiments/determine_factor_against_truth/sx000x", exist_ok=True)
        plt.savefig(f"./results/experiments/determine_factor_against_truth/sx000x/{metric}_over_factor.pdf", bbox_inches="tight")
        plt.cla()

        recorddata = np.array(truth_vel_plot_records, dtype=[
            ("Experiment Name", f"U{len('s30001')}"),
            ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
            ("Segment Name", f"U{len('Upper Arm Right')}"),
            ("Factor", "f"),
            ("RMSE", "f"),
            ("DTW", "f"),
            ("DFD", "f"),
            ("PCC", "f"),
        ])
        dataframe = pd.DataFrame.from_records(recorddata)
        sns.set_style("darkgrid")
        len_joints = len(MATCHING_JOINTS)
        linestyles=(["-"] * 8 + [":"] * 8)[:len_joints] + ["-."]
        ax = sns.catplot(
            data=dataframe,
            x="Factor",
            y=metric,
            kind="point",
            hue="Filter Name",
            linestyles=linestyles,
            markersize=0,
            linewidth=1,
            native_scale=True,
            errorbar=None
        )
        # ax.set_xticklabels(rotation=40, ha="right")
        # ticks = ax.ax.get_xticklabels()
        # plt.setp(ax.ax.get_xticklabels(), visible=False)
        # plt.setp(ticks[::6], visible=True)
        plt.xlabel(r"$\lambda$")
        # Otherwise take default
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(rf"Mean {metric} for different $\lambda$ Evaluated on Joint Velocities Against Ground Truth")
        os.makedirs(f"./results/experiments/determine_factor_against_truth/sx000x", exist_ok=True)
        plt.savefig(f"./results/experiments/determine_factor_against_truth/sx000x/{metric}_over_factor_vel.pdf", bbox_inches="tight")
        plt.cla()





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder")
    parser.add_argument("-t", "--experiment-type", dest="experiment_type", choices=["cop", "cop-wide", "constraint", "constraint-fast"])
    parser.add_argument("-x", "--early-exit", dest="early_exit", action="store_true")
    parser.add_argument("-s", "--show", dest="show", action="store_true", default=False)
    parser.add_argument("-c", "--compare", dest="compare", action="store_true", default=False)
    parser.add_argument("-f", "--second-folder", dest="second_folder")
    parser.add_argument("-p", dest="predictions", action="store_true", default=False)
    parser.add_argument("-e", "--experiment_name", dest="experiment_name")
    parser.add_argument("-v", dest="vel", action="store_true", default=False)
    parser.add_argument("-l", "--latex", dest="latextables", action="store_true", default=False)
    parser.add_argument("-j", "--jointlenght", dest="jointlength", action="store_true", default=False)
    parser.add_argument("-b", "--blandaltman", dest="blandaltman", action="store_true", default=False)
    parser.add_argument("-z", "--benchmark", dest="benchmark", action="store_true", default=False)

    args = parser.parse_args()

    global SHOW
    SHOW = args.show

    global FILTER_NAME
    FILTER_NAME = os.path.basename(os.path.dirname(args.experiment_folder))

    os.makedirs(f"./results/experiments/{FILTER_NAME}/", exist_ok=True)
    os.makedirs(f"./results/experiments/{FILTER_NAME}/determine_factor/", exist_ok=True)
    os.makedirs(f"./results/experiments/{FILTER_NAME}/determine_factor_against_truth/", exist_ok=True)
    os.makedirs(f"./results/experiments/{FILTER_NAME}/joint_segment_lengths/", exist_ok=True)
    os.makedirs(f"./results/experiments/{FILTER_NAME}/joint_trajectories/", exist_ok=True)
    os.makedirs(f"./results/experiments/{FILTER_NAME}/joint_velocities/", exist_ok=True)
    os.makedirs(f"./results/experiments/{FILTER_NAME}/cop_trajectories/", exist_ok=True)
    os.makedirs(f"./results/experiments/{FILTER_NAME}/subplots/", exist_ok=True)

    ex_name = os.path.basename(args.experiment_folder)

    if args.latextables:
        all_ex_find_best_measurement_error_factor_rmse(Path(args.experiment_folder))
        return

    if args.predictions:
        if "s3" in args.experiment_name:
            compare_prediction_vs_truth_for_different_filters(Path(args.experiment_folder), vel=False)
            compare_prediction_vs_truth_for_different_filters(Path(args.experiment_folder), vel=True)
        elif "s1" in args.experiment_name:
            compare_prediction_vs_truth_for_different_filters_qtm_for_com(Path(args.experiment_folder), args.experiment_name, vel=False)
            compare_prediction_vs_truth_for_different_filters_qtm_for_com(Path(args.experiment_folder), args.experiment_name, vel=True)
        else:
            raise ValueError(f"{ex_name} not supported")

        return

    if args.jointlength:
        create_joint_length_plots_and_table(Path(args.experiment_folder))
        return

    if args.blandaltman:
        for filter_type in tqdm(FILTER_TYPES):
            for ex_name in tqdm(EX_NAMES):
                best_factor = 35
                FILTER_NAME = filter_type
                path = Path(args.experiment_folder) / filter_type / ex_name
                if "s3" in ex_name:
                    theia_data = load_processed_theia_data(find_factor_path(best_factor, path))
                    bland_altman_plots(theia_data, ex_name)
                    plot_subparts_of_trajectories(theia_data, ex_name)
                elif "s1" in ex_name:
                    qtm_data = load_processed_data(find_factor_path(best_factor, path))
                    bland_altman_plots_qtm(qtm_data, ex_name)
                else:
                    raise ValueError(f"Unknown experiment_name: {ex_name}")

        return

    if args.benchmark:
        compare_frames(Path(args.experiment_folder))
        return

    #if args.second_folder:
    #    determine_minimum_against_ground_truth(args)

    '''
    cutoff = 0.01
    vel_result = compare_qtm_joints_kinect_joints_vel(data, cutoff)
    '''

    axss = ["X", "Y", "Z"]
    if args.compare:
        joint = int(Joint.HEAD)
        idx = 2
        for error in [25]:
            data1 = cond_load_data(find_factor_path(error, Path(args.experiment_folder)))
            data2 = cond_load_data(find_factor_path(error, Path(args.second_folder)))

            a = data1.kinect_joints[:, int(joint), idx]
            b = data2.kinect_joints[:, int(joint), idx]

            plt.plot(data1.kinect_ts, a, "-", label=f"Without acceleration - lambda: {error}", alpha=0.8)
            plt.plot(data2.kinect_ts, b, label=f"With acceleration - lambda: {error}", alpha=0.8)

        data1 = cond_load_data(find_factor_path(0, Path(args.experiment_folder)))
        a = data1.down_kinect_unfiltered_joints[:, int(joint), idx]
        plt.plot(data1.down_kinect_ts, double_butter(a), "--", label=f"Raw", alpha=0.8)

        plt.xlabel("Time [s]")
        plt.ylabel(f"Axis {axss[idx]} [m]")
        plt.title(j2str(joint))
        plt.legend()
        plt.show()

        return

    cutoff = 0.01
    # path, factor = find_best_measurement_error_factor_corr_on_velocity(Path(args.experiment_folder), cutoff, args.experiment_type)
    theia = "s300" in ex_name
    if theia:
        args.experiment_type = "constraint"
    else:
        data = load_processed_data(Path(args.experiment_folder) / "0")
        print(f"Ex: {ex_name}")
        print(f"Var X: {data.qtm_cop[:, 0].var()}")
        print(f"Var Y: {data.qtm_cop[:, 1].var()}")
        if data.qtm_cop[:, 0].var() > 0.001 and data.qtm_cop[:, 1].var() > 0.001:
            args.experiment_type = "cop"
        else:
            args.experiment_type = "constraint"

    print(f"Extracted experiment type: {args.experiment_type}")

    '''
    pool = ThreadPool(processes=2)
    joint_result = pool.apply_async(find_best_measurement_error_factor_rmse, (Path(args.experiment_folder), cutoff, args.experiment_type))
    vel_result = pool.apply_async(find_best_measurement_error_factor_rmse_on_velocity, (Path(args.experiment_folder), cutoff, args.experiment_type))

    joint_path, joint_rmse_factor, joint_dtw_factor, joint_fr_factor = joint_result.get()
    vel_path, vel_rmse_factor, vel_dtw_factor, vel_fr_factor = vel_result.get()
    '''

    joint_path, joint_rmse_factor, joint_dtw_factor, joint_fr_factor, joint_pcc_factor, record = find_best_measurement_error_factor_rmse(Path(args.experiment_folder), ex_name, cutoff)
    vel_path, vel_rmse_factor, vel_dtw_factor, vel_fr_factor, vel_pcc_factor, vel_record = find_best_measurement_error_factor_rmse_on_velocity(Path(args.experiment_folder), ex_name, cutoff)



    print(f"Ex: {map_ex_name(ex_name)}")
    print("Velocity")
    print(f"rmse factor: {vel_rmse_factor}")
    print(f"dtw factor: {vel_dtw_factor}")
    print(f"fr factor: {vel_fr_factor}")

    print("Joints")
    print(f"rmse factor: {joint_rmse_factor}")
    print(f"dtw factor: {joint_dtw_factor}")
    print(f"fr factor: {joint_fr_factor}")
    # data = load_processed_data(vel_path)
    best_factor = (((joint_fr_factor + vel_rmse_factor + vel_dtw_factor + vel_fr_factor) / 3 ) // 5 ) * 5
    best_factor = 80
    print(f"best factor: {best_factor}")

    if theia:
        determine_minimum_against_ground_truth_theia(Path(args.experiment_folder), ex_name)
        theia_data = load_processed_theia_data(find_factor_path(best_factor, Path(args.experiment_folder)))
        for (kinect_joints, theia_joints, name) in JOINT_SEGMENTS:
            print(f"Segment: {name}")
            results = compare_theia_joints_kinect_joints(theia_data, kinect_joints, theia_joints)
            print("Theia Joint Results:")
            print(results)

            results = compare_theia_joints_kinect_joints_vel(theia_data, kinect_joints, theia_joints)
            print("Theia Joint Vel Results:")
            print(results)

        results = compare_theia_joints_kinect_com(theia_data)
        print("Theia COM Results:")
        print(results)

        results = compare_theia_joints_kinect_com_vel(theia_data)
        print("Theia COM Vel Results:")
        print(results)

        # plot_constrained_segment_joint_length_change(ex_name, theia_data, cutoff)

        plot_subparts_of_trajectories(theia_data, ex_name)
        bland_altman_plots(theia_data, ex_name)
        return

    data = load_processed_data(find_factor_path(best_factor, Path(args.experiment_folder)))

    joint_result = None
    vel_result = None
    joint_result_inverted_right = None
    vel_result_inverted_right = None
    if args.experiment_type in ["cop", "cop-wide"]:
        joint_result = compare_qtm_cop_kinect_cop(data, cutoff)
        vel_result = compare_qtm_cop_kinect_cop_vel(data, cutoff)
    else:
        joint_result_inverted_right = compare_qtm_joints_kinect_joints_inverted_right(data, cutoff)
        vel_result_inverted_right = compare_qtm_joints_kinect_joints_vel_inverted_right(data, cutoff)
        joint_result = compare_qtm_joints_kinect_joints(data, cutoff)
        vel_result = compare_qtm_joints_kinect_joints_vel(data, cutoff)

    plot_constrained_segment_joint_length_change(ex_name, data, cutoff)
    print("Joints")
    print("Left")
    print(joint_result)
    print("Inverted Right")
    print(joint_result_inverted_right)

    print("Velocities")
    print("Left")
    print(vel_result)
    print("Inverted Right")
    print(vel_result_inverted_right)


    factors = [5, 50, best_factor]
    datas = [load_processed_data(find_factor_path(factor, Path(args.experiment_folder))) for factor in factors]

    best_factors = [best_factor]
    best_datas = [load_processed_data(find_factor_path(best_factor, Path(args.experiment_folder)))]

    if args.experiment_type in ["cop", "cop-wide"]:
        plot_cop_x_y_for_different_factors(ex_name, factors, datas, cutoff)
        plot_cop_x_y_for_different_factors(ex_name, best_factors, best_datas, cutoff, "best")
    else:
        plot_joints_for_different_factors(ex_name, factors, datas, cutoff)
        plot_velocities_for_different_factors(ex_name, factors, datas, cutoff)

        plot_joints_for_different_factors(ex_name, best_factors, best_datas, cutoff, "best")
        plot_velocities_for_different_factors(ex_name, best_factors, best_datas, cutoff, "best")

    # compare_velocities(load_processed_data(Path(args.experiment_folder) / "6"))
    compare_velocities(data)

    if args.early_exit:
        return

    if args.experiment_type in ["cop", "cop-wide"]:
        plt.cla()
        o = int(data.down_kinect_com.shape[0] * cutoff)
        fig, ax = plt.subplots(1, 2)
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
        fig.suptitle("Compare X and Y Axis of COP and COM")

        plt.show()


    else:
        # Manual check to see if the butterworth artifacts have been cutoff
        o = int(data.down_kinect_joints.shape[0] * cutoff)
        plt.plot(data.down_kinect_ts[o:-o], data.down_kinect_joints[:, int(Joint.ELBOW_LEFT), 2][o:-o], label="Kalman Filtered")
        plt.plot(data.down_kinect_ts[o:-o], double_butter(data.down_kinect_unfiltered_joints[:, int(Joint.ELBOW_LEFT), 2])[o:-o], label="Double Butterworth Filtered")
        plt.plot(data.down_kinect_ts[o:-o], data.down_kinect_unfiltered_joints[:, int(Joint.ELBOW_LEFT), 2][o:-o], label="Raw Data")
        plt.legend()
        plt.title("Compare Joint Position ELBOW_LEFT for Z Axis")
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

def compare_theia_joints_kinect_com_vel(data: TheiaData, cutoff: float = 0.15) -> tuple[float, float, float, float, float, float, float, float]:
    theia_joint = int(TheiaJoint.COM_VEL)

    corr = 0
    corr_un = 0

    rmse = 0
    rmse_un = 0

    dtw_dist = 0
    dtw_dist_un = 0

    fr_dist = 0
    fr_dist_un = 0

    theia_ts = np.arange(data.theia_tensor.shape[0]) * (1./120.)

    down_theia = downsample(double_butter(data.theia_tensor[:, theia_joint, :], 120), theia_ts, 15)
    length = min(down_theia.shape[0], data.down_kinect_joints.shape[0])
    o = max(int(length * cutoff), 1)

    down_theia = down_theia[:length][o:-o]

    vel = data.down_kinect_com_velocities[:length][o:-o]
    vel_un = central_diff(double_butter(data.down_kinect_unfiltered_com), 15)[:length][o:-o]

    corr += np.corrcoef(down_theia[:, 0], vel[:, 0])[0, 1] + np.corrcoef(down_theia[:, 1], vel[:, 1])[0, 1] + np.corrcoef(down_theia[:, 2], vel[:, 2])[0, 1]
    corr_un += np.corrcoef(down_theia[:, 0], vel_un[:, 0])[0, 1] + np.corrcoef(down_theia[:, 1], vel_un[:, 1])[0, 1] + np.corrcoef(down_theia[:, 2], vel_un[:, 2])[0, 1]

    diff = np.linalg.norm(down_theia - vel, axis=1)
    rmse += np.sqrt(np.mean(np.power(diff, 2)))

    diff_un = np.linalg.norm(down_theia - vel_un, axis=1)
    rmse_un += np.sqrt(np.mean(np.power(diff_un, 2)))

    dtw_dist += dtw.dtw(down_theia, vel, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
    dtw_dist_un += dtw.dtw(down_theia, vel_un, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

    fr_dist += frechet_dist(down_theia, vel)
    fr_dist_un += frechet_dist(down_theia, vel_un)

    # correlation is 3 axis
    return corr / (3), corr_un / (3), rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un


def compare_theia_joints_kinect_com(data: TheiaData, cutoff: float = 0.15) -> tuple[float, float, float, float, float, float, float, float]:
    theia_joint = int(TheiaJoint.COM)

    corr = 0
    corr_un = 0

    rmse = 0
    rmse_un = 0

    dtw_dist = 0
    dtw_dist_un = 0

    fr_dist = 0
    fr_dist_un = 0

    theia_ts = np.arange(data.theia_tensor.shape[0]) * (1./120.)

    down_theia = downsample(double_butter(data.theia_tensor[:, theia_joint, :], 120), theia_ts, 15)
    length = min(down_theia.shape[0], data.down_kinect_joints.shape[0])
    o = max(int(length * cutoff), 1)

    down_theia = down_theia[:length][o:-o]

    joints = data.down_kinect_com[:length][o:-o]
    joints_un = double_butter(data.down_kinect_unfiltered_com)[:length][o:-o]

    corr += np.corrcoef(down_theia[:, 0], joints[:, 0])[0, 1] + np.corrcoef(down_theia[:, 1], joints[:, 1])[0, 1] + np.corrcoef(down_theia[:, 2], joints[:, 2])[0, 1]
    corr_un += np.corrcoef(down_theia[:, 0], joints_un[:, 0])[0, 1] + np.corrcoef(down_theia[:, 1], joints_un[:, 1])[0, 1] + np.corrcoef(down_theia[:, 2], joints[:, 2])[0, 1]

    diff = np.linalg.norm(down_theia - joints, axis=1)
    rmse += np.sqrt(np.mean(np.power(diff, 2)))

    diff_un = np.linalg.norm(down_theia - joints_un, axis=1)
    rmse_un += np.sqrt(np.mean(np.power(diff_un, 2)))

    dtw_dist += dtw.dtw(down_theia, joints, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
    dtw_dist_un += dtw.dtw(down_theia, joints_un, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

    fr_dist += frechet_dist(down_theia, joints)
    fr_dist_un += frechet_dist(down_theia, joints_un)

    # correlation is 3 joints and 3 axis
    return corr / (3*3), corr_un / (3*3), rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un


def compare_theia_joints_kinect_joints_vel(data: TheiaData, kinect_joints_: list[Joint], theia_joints_: list[TheiaJoint], cutoff: float = 0.15) -> tuple[float, float, float, float, float, float, float, float]:
    kinect_joints = [int(element) for element in [Joint.SHOULDER_RIGHT, Joint.ELBOW_RIGHT, Joint.WRIST_RIGHT]]
    theia_joints = [int(element) for element in [TheiaJoint.SHOULDER_RIGHT, TheiaJoint.ELBOW_RIGHT, TheiaJoint.WRIST_RIGHT]]

    corr = 0
    corr_un = 0

    rmse = 0
    rmse_un = 0

    dtw_dist = 0
    dtw_dist_un = 0

    fr_dist = 0
    fr_dist_un = 0

    theia_ts = np.arange(data.theia_tensor.shape[0]) * (1./120.)

    for kinect_joint, theia_joint in zip(kinect_joints, theia_joints):
        down_theia = downsample(central_diff(double_butter(data.theia_tensor[:, theia_joint, :], 120), 120), theia_ts, 15)
        length = min(down_theia.shape[0], data.down_kinect_joints.shape[0])
        o = max(int(length * cutoff), 1)

        down_theia = down_theia[:length][o:-o]

        vel = data.down_kinect_velocities[:, kinect_joint,:][:length][o:-o]
        vel_un = central_diff(double_butter(data.down_kinect_unfiltered_joints[:, kinect_joint, :]), 15)[:length][o:-o]

        corr += np.corrcoef(down_theia[:, 0], vel[:, 0])[0, 1] + np.corrcoef(down_theia[:, 1], vel[:, 1])[0, 1] + np.corrcoef(down_theia[:, 2], vel[:, 2])[0, 1]
        corr_un += np.corrcoef(down_theia[:, 0], vel_un[:, 0])[0, 1] + np.corrcoef(down_theia[:, 1], vel_un[:, 1])[0, 1] + np.corrcoef(down_theia[:, 2], vel_un[:, 2])[0, 1]

        diff = np.linalg.norm(down_theia - vel, axis=1)
        rmse += np.sqrt(np.mean(np.power(diff, 2)))

        diff_un = np.linalg.norm(down_theia - vel_un, axis=1)
        rmse_un += np.sqrt(np.mean(np.power(diff_un, 2)))

        dtw_dist += dtw.dtw(down_theia, vel, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
        dtw_dist_un += dtw.dtw(down_theia, vel_un, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

        fr_dist += frechet_dist(down_theia, vel)
        fr_dist_un += frechet_dist(down_theia, vel_un)

    # correlation is 3 joints and 3 axis
    return corr / (3*3), corr_un / (3*3), rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un

def compare_theia_joints_kinect_joints(data: TheiaData, kinect_joints_: list[Joint], theia_joints_: list[TheiaJoint], cutoff: float = 0.15) -> tuple[float, float, float, float, float, float, float, float]:
    kinect_joints = [int(element) for element in kinect_joints_]
    theia_joints = [int(element) for element in theia_joints_]

    corr = 0
    corr_un = 0

    rmse = 0
    rmse_un = 0

    dtw_dist = 0
    dtw_dist_un = 0

    fr_dist = 0
    fr_dist_un = 0

    theia_ts = np.arange(data.theia_tensor.shape[0]) * (1./120.)

    for kinect_joint, theia_joint in zip(kinect_joints, theia_joints):
        down_theia = downsample(double_butter(data.theia_tensor[:, theia_joint, :], 120), theia_ts, 15)
        length = min(down_theia.shape[0], data.down_kinect_joints.shape[0])
        o = max(int(length * cutoff), 1)

        down_theia = down_theia[:length][o:-o]

        joints = data.down_kinect_joints[:, kinect_joint,:][:length][o:-o]
        joints_un = double_butter(data.down_kinect_unfiltered_joints[:, kinect_joint, :])[:length][o:-o]

        corr += np.corrcoef(down_theia[:, 0], joints[:, 0])[0, 1] + np.corrcoef(down_theia[:, 1], joints[:, 1])[0, 1] + np.corrcoef(down_theia[:, 2], joints[:, 2])[0, 1]
        corr_un += np.corrcoef(down_theia[:, 0], joints_un[:, 0])[0, 1] + np.corrcoef(down_theia[:, 1], joints_un[:, 1])[0, 1] + np.corrcoef(down_theia[:, 2], joints[:, 2])[0, 1]

        diff = np.linalg.norm(down_theia - joints, axis=1)
        rmse += np.sqrt(np.mean(np.power(diff, 2)))

        diff_un = np.linalg.norm(down_theia - joints_un, axis=1)
        rmse_un += np.sqrt(np.mean(np.power(diff_un, 2)))

        dtw_dist += dtw.dtw(down_theia, joints, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance
        dtw_dist_un += dtw.dtw(down_theia, joints_un, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

        fr_dist += frechet_dist(down_theia, joints)
        fr_dist_un += frechet_dist(down_theia, joints_un)

    # correlation is 3 joints and 3 axis
    return corr / (3*3), corr_un / (3*3), rmse, rmse_un, dtw_dist, dtw_dist_un, fr_dist, fr_dist_un


def plot_subparts_of_trajectories(theia_data: TheiaData, ex_name: str) -> None:
    length = theia_data.min_joint_length_at_15hz

    unfiltered = theia_data.down_kinect_unfiltered_joints[:length]
    filtered = theia_data.down_kinect_joints[:length]
    truth = downsample(theia_data.theia_tensor, np.arange(theia_data.theia_tensor.shape[0]) * (1./120.), 15)[:length]

    vel_unfiltered = central_diff(theia_data.down_kinect_unfiltered_joints[:length], 15)
    vel_filtered = theia_data.down_kinect_velocities[:length]
    vel_truth = downsample(theia_data.theia_tensor, np.arange(theia_data.theia_tensor.shape[0]) * (1./120.), 15)[:length]

    time = np.arange(0, length) * (1./15.)

    step = 2

    start_end = (time[-1] // step) * step
    starts = np.arange(step, start_end-step, step)
    ends = np.arange(3*step, start_end+1, step)
    timeslots = np.column_stack((starts, ends))

    for kinect_joint, theia_joint, joint_name in tqdm(MATCHING_JOINTS):


        for ax_idx, ax_name in enumerate(["X", "Y", "Z"]):
            for slot in timeslots:
                ts = np.linspace(slot[0], slot[1], 2 * step * 15)
                m, n = int(slot[0] * 15), int(slot[1] * 15)
                plt.cla()
                plt.plot(ts, unfiltered[m:n, int(kinect_joint), ax_idx], label="Unfiltered Kinect")
                plt.plot(ts, filtered[m:n, int(kinect_joint), ax_idx], label="Filtered Kinect")
                plt.plot(ts, truth[m:n, int(theia_joint), ax_idx], label="Theia")
                plt.legend()
                plt.xlabel(f"{slot[0]} - {slot[1]} [s]")
                plt.ylabel(f"{ax_name} [m]")
                os.makedirs(f"./results/experiments/{FILTER_NAME}/subplots/{ex_name}/{joint_name}/{ax_name}/", exist_ok=True)
                plt.savefig(f"./results/experiments/{FILTER_NAME}/subplots/{ex_name}/{joint_name}/{ax_name}/{slot[0]}-{slot[1]}.pdf", bbox_inches="tight")
                plt.cla()

                plt.cla()
                plt.plot(ts, vel_unfiltered[m:n, int(kinect_joint), ax_idx], label="Unfiltered Kinect")
                plt.plot(ts, vel_filtered[m:n, int(kinect_joint), ax_idx], label="Filtered Kinect")
                plt.plot(ts, vel_truth[m:n, int(theia_joint), ax_idx], label="Theia")
                plt.legend()
                plt.xlabel(f"{slot[0]} - {slot[1]} [s]")
                plt.ylabel(f"{ax_name} [m]")
                os.makedirs(f"./results/experiments/{FILTER_NAME}/subplots/{ex_name}/{joint_name}/{ax_name}/vel", exist_ok=True)
                plt.savefig(f"./results/experiments/{FILTER_NAME}/subplots/{ex_name}/{joint_name}/{ax_name}/vel/{slot[0]}-{slot[1]}.pdf", bbox_inches="tight")
                plt.cla()


    vel_unfiltered = central_diff(theia_data.down_kinect_unfiltered_com[:length], 15)
    vel_filtered = theia_data.down_kinect_velocities[:length]
    vel_truth = central_diff(truth[:length, TheiaJoint.COM, :], 15)

    unfiltered = theia_data.down_kinect_unfiltered_com[:length]
    filtered = theia_data.down_kinect_com[:length]
    truth = truth[:length, TheiaJoint.COM, :]

    joint_name = "CoM"
    for ax_idx, ax_name in enumerate(["X", "Y", "Z"]):
        for slot in timeslots:
            ts = np.linspace(slot[0], slot[1], 2 * step * 15)
            m, n = int(slot[0] * 15), int(slot[1] * 15)
            plt.cla()
            plt.plot(ts, unfiltered[m:n, ax_idx], label="Unfiltered Kinect")
            plt.plot(ts, filtered[m:n, ax_idx], label="Filtered Kinect")
            plt.plot(ts, truth[m:n, ax_idx], label="Theia")
            plt.legend()
            plt.xlabel(f"{slot[0]} - {slot[1]} [s]")
            plt.ylabel(f"{ax_name} [m]")
            os.makedirs(f"./results/experiments/{FILTER_NAME}/subplots/{ex_name}/{joint_name}/{ax_name}/", exist_ok=True)
            plt.savefig(f"./results/experiments/{FILTER_NAME}/subplots/{ex_name}/{joint_name}/{ax_name}/{slot[0]}-{slot[1]}.pdf", bbox_inches="tight")
            plt.cla()

            plt.cla()
            plt.plot(ts, unfiltered[m:n, ax_idx], label="Unfiltered Kinect")
            plt.plot(ts, filtered[m:n, ax_idx], label="Filtered Kinect")
            plt.plot(ts, truth[m:n, ax_idx], label="Theia")
            plt.legend()
            plt.xlabel(f"{slot[0]} - {slot[1]} [s]")
            plt.ylabel(f"{ax_name} [m]")
            os.makedirs(f"./results/experiments/{FILTER_NAME}/subplots/{ex_name}/{joint_name}/{ax_name}/vel", exist_ok=True)
            plt.savefig(f"./results/experiments/{FILTER_NAME}/subplots/{ex_name}/{joint_name}/{ax_name}/vel/{slot[0]}-{slot[1]}.pdf", bbox_inches="tight")
            plt.cla()

def bland_altman_plots_qtm(qtm_data: Data, ex_name: str, cutoff: float = 0.20) -> None:
    joint_name = "CoP"
    os.makedirs(f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/", exist_ok=True)

    unfiltered = qtm_data.down_kinect_unfiltered_com
    filtered = qtm_data.down_kinect_com
    truth = downsample(qtm_data.qtm_cop, qtm_data.qtm_cop_ts, 15)
    length = min(len(unfiltered), len(filtered), len(truth))

    o = max(int(length * cutoff), 1)

    unfiltered = unfiltered[:length][o:-o]
    filtered = filtered[:length][o:-o]
    truth = truth[:length][o:-o]

    for ax_idx, ax_name in enumerate(["X", "Y"]):
        path = f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/{joint_name}_{ax_name}_filtered.pdf"
        with Path(path).open(mode="wb+") as _f:
            bland_altman(filtered[:, ax_idx], truth[:, ax_idx], _f)
        path = f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/{joint_name}_{ax_name}_unfiltered.pdf"
        with Path(path).open(mode="wb+") as _f:
            bland_altman(unfiltered[:, ax_idx], truth[:, ax_idx], _f)



def bland_altman_plots(theia_data: TheiaData, ex_name: str, cutoff: float = 0.20) -> None:
    length = theia_data.min_joint_length_at_15hz
    o = max(int(length * cutoff), 1)

    os.makedirs(f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/", exist_ok=True)

    unfiltered = theia_data.down_kinect_unfiltered_joints[:length][o:-o]
    filtered = theia_data.down_kinect_joints[:length][o:-o]
    truth = downsample(theia_data.theia_tensor, np.arange(theia_data.theia_tensor.shape[0]) * (1./120.), 15)[:length][o:-o]
    time = np.arange(0, length) * (1./15.)

    for kinect_joint, theia_joint, joint_name in tqdm(MATCHING_JOINTS):
        for ax_idx, ax_name in enumerate(["X", "Y", "Z"]):
            path = f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/{joint_name}_{ax_name}_filtered.pdf"
            with Path(path).open(mode="wb+") as _f:
                bland_altman(filtered[:, int(kinect_joint), ax_idx], truth[:, int(theia_joint), ax_idx], _f)
            path = f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/{joint_name}_{ax_name}_unfiltered.pdf"
            with Path(path).open(mode="wb+") as _f:
                bland_altman(unfiltered[:, int(kinect_joint), ax_idx], truth[:, int(theia_joint), ax_idx], _f)

    unfiltered = theia_data.down_kinect_unfiltered_com[:length][o:-o]
    filtered = theia_data.down_kinect_com[:length][o:-o]
    truth = truth[:length, TheiaJoint.COM, :]

    joint_name = "CoM"
    for ax_idx, ax_name in enumerate(["X", "Y", "Z"]):
        path = f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/{joint_name}_{ax_name}_filtered.pdf"
        with Path(path).open(mode="wb+") as _f:
            bland_altman(filtered[:, ax_idx], truth[:, ax_idx], _f)
        path = f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/{joint_name}_{ax_name}_unfiltered.pdf"
        with Path(path).open(mode="wb+") as _f:
            bland_altman(unfiltered[:, ax_idx], truth[:, ax_idx], _f)

    # XCOM
    joint_name = "XcoM"
    est = theia_data.down_kinect_xcom[:length][o:-o]
    un = theia_data.down_kinect_unfiltered_xcom[:length][o:-o]
    tru = theia_data.down_theia_xcom_15_hz[:length][o:-o]
    for ax_idx, ax_name in enumerate(["X", "Y"]):
        path = f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/{joint_name}_{ax_name}_filtered.pdf"
        with Path(path).open(mode="wb+") as _f:
            bland_altman(est[:, ax_idx], tru[:, ax_idx], _f)
        path = f"./results/experiments/{FILTER_NAME}/bland_altman/{ex_name}/{joint_name}_{ax_name}_unfiltered.pdf"
        with Path(path).open(mode="wb+") as _f:
            bland_altman(un[:, ax_idx], tru[:, ax_idx], _f)


def compare_filter_type(experiment_path_a: Path, experiment_path_b: Path) -> None:
    pass

def compare_prediction_vs_truth_for_different_filters(experiment_path: Path, cutoff: float = 0.1, vel: bool = False) -> None:
    # for filter_name in ["SimpleSkeletonFilter"]:
    records = []
    com_records = []
    xcom_records = []
    # for ex_name in tqdm([f"s3000{i}" for i in range(1, 2)]):
    for ex_name in tqdm([f"s3000{i}" for i in range(1, 7)]):
        for best_factor in tqdm(np.arange(0, 150, 5)):
            for filter_name in ["ConstrainedSkeletonFilter", "SkeletonFilter", "SimpleConstrainedSkeletonFilter", "SimpleSkeletonFilter"]:
                path = experiment_path / filter_name / ex_name
                data = load_processed_theia_data(find_factor_path(best_factor, path))

                """
                if best_factor > 50:
                    plt.plot(data.down_kinect_xcom[:, 0], label="Kinect X")
                    plt.plot(data.down_theia_xcom_15_hz[:, 0], label="Theia X")
                    plt.legend()
                    plt.show()
                    plt.cla()
                """

                length = data.min_joint_length_at_15hz
                o = max(int(length * cutoff), 1)

                prediction = data.down_kinect_predictions[:length][o:-o]
                estimate = None
                if not vel:
                    estimate = data.down_kinect_joints[:length][o:-o]
                else:
                    estimate = data.down_kinect_velocities[:length][o:-o]
                unfiltered = data.down_kinect_unfiltered_joints[:length][o:-o]
                truth = downsample(double_butter(data.theia_tensor, 120), np.arange(data.theia_tensor.shape[0]) * (1./120.), 15)[:length][o:-o]

                l_pred_to_est_rmse = []
                l_truth_to_pred_rmse = []
                l_truth_to_est_rmse = []
                l_truth_to_unfiltered_rmse = []

                l_truth_to_est_fr = []
                l_truth_to_un_fr = []

                l_truth_to_est_dtw = []
                l_truth_to_un_dtw = []

                l_truth_to_est_pcc = []
                l_truth_to_un_pcc = []

                for kinect_joint, theia_joint, joint_name in MATCHING_JOINTS:
                    kinect_joint, theia_joint = int(kinect_joint), int(theia_joint)


                    pred = None
                    est = None
                    un = None
                    tru = None
                    est = estimate[:, kinect_joint, :]
                    pred = prediction[:, kinect_joint, :]
                    if not vel:
                        un = unfiltered[:, kinect_joint, :]
                        tru = truth[:, theia_joint, :]
                    else:
                        un = central_diff(unfiltered[:, kinect_joint, :], 15)
                        tru = central_diff(truth[:, theia_joint, :], 15)

                    # sun, sutru = corr_shift_trim3d(un, tru)
                    # sest, stru = corr_shift_trim3d(est, tru)

                    sun, sutru = un, tru
                    sest, stru = est, tru


                    '''
                    for i in range(3):
                        plt.plot(un[:, i], label="un")
                        plt.plot(tru[:, i], label="true")
                        plt.plot(shift_est[:, i], label="shift est")
                        # plt.plot(est[:, i], label="est")
                        plt.legend()
                        plt.show()
                        plt.cla()
                        '''

                    diff = np.linalg.norm(pred - est, axis=1)
                    pred_to_est_rmse = np.sqrt(np.mean(np.power(diff, 2)))

                    diff = np.linalg.norm(tru - pred, axis=1)
                    truth_to_pred_rmse = np.sqrt(np.mean(np.power(diff, 2)))

                    diff = np.linalg.norm(stru - sest, axis=1)
                    truth_to_est_rmse = np.sqrt(np.mean(np.power(diff, 2)))

                    diff = np.linalg.norm(sutru - sun, axis=1)
                    truth_to_un_rmse = np.sqrt(np.mean(np.power(diff, 2)))

                    truth_to_un_pcc = (np.corrcoef(sutru[:, 0], sun[:, 0])[0, 1] + np.corrcoef(sutru[:, 1], sun[:, 1])[0, 1] + np.corrcoef(sutru[:, 2], sun[:, 2])[0, 1]) / 3
                    truth_to_est_pcc = (np.corrcoef(stru[:, 0], sest[:, 0])[0, 1] + np.corrcoef(stru[:, 1], sest[:, 1])[0, 1] + np.corrcoef(stru[:, 2], sest[:, 2])[0, 1]) / 3

                    p = np.column_stack((np.arange(len(stru)), stru))
                    q = np.column_stack((np.arange(len(sest)), sest))
                    truth_to_est_fr = frechet_dist(p, q)
                    truth_to_est_dtw = dtw.dtw(p, q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

                    p = np.column_stack((np.arange(len(sutru)), sutru))
                    q = np.column_stack((np.arange(len(sun)), sun))
                    truth_to_un_fr = frechet_dist(p, q)
                    truth_to_un_dtw = dtw.dtw(p, q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

                    # print(f"Filter: {filter_name} Joint: {joint_name}")
                    # print(f"pred_to_est_rmse: {pred_to_est_rmse}")
                    # print(f"truth_to_pred_rmse: {truth_to_pred_rmse}")
                    # print(f"truth_to_est_rmse: {truth_to_est_rmse}")
                    # print(f"truth_to_un_rmse: {truth_to_un_rmse}")

                    l_pred_to_est_rmse.append(pred_to_est_rmse)
                    l_truth_to_pred_rmse.append(truth_to_pred_rmse)
                    l_truth_to_est_rmse.append(truth_to_est_rmse)
                    l_truth_to_unfiltered_rmse.append(truth_to_un_rmse)

                    l_truth_to_est_fr.append(truth_to_est_fr)
                    l_truth_to_un_fr.append(truth_to_un_fr)

                    l_truth_to_est_dtw.append(truth_to_est_dtw)
                    l_truth_to_un_dtw.append(truth_to_un_dtw)

                    l_truth_to_est_pcc.append(truth_to_est_pcc)
                    l_truth_to_un_pcc.append(truth_to_un_pcc)

                    record = (
                        ex_name,
                        short_name(filter_name),
                        joint_name,
                        best_factor,
                        truth_to_est_rmse,
                        truth_to_est_dtw,
                        truth_to_est_fr,
                        truth_to_est_pcc,
                    )
                    records.append(record)

                record = (
                    ex_name,
                    short_name(filter_name),
                    "Mean",
                    best_factor,
                    np.array(l_truth_to_est_rmse).mean(),
                    np.array(l_truth_to_est_dtw).mean(),
                    np.array(l_truth_to_est_fr).mean(),
                    np.array(l_truth_to_est_pcc).mean(),
                )
                records.append(record)

                # COM
                est = None
                un = None
                tru = None
                if not vel:
                    est = data.down_kinect_com[:length][o:-o]
                    un = data.down_kinect_unfiltered_com[:length][o:-o]
                    tru = truth[:, int(TheiaJoint.COM), :]
                else:
                    est = data.down_kinect_com_velocities[:length][o:-o]
                    un = central_diff(data.down_kinect_unfiltered_com[:length][o:-o], 15)
                    tru = central_diff(double_butter(truth[:, int(TheiaJoint.COM), :]), 15)

                # sun, sutru = corr_shift_trim3d(un, tru)
                # sest, stru = corr_shift_trim3d(est, tru)

                sun, sutru = un, tru
                sest, stru = est, tru

                diff = np.linalg.norm(stru - sest, axis=1)
                truth_to_est_rmse_com = np.sqrt(np.mean(np.power(diff, 2)))

                diff = np.linalg.norm(sutru - sun, axis=1)
                truth_to_un_com = np.sqrt(np.mean(np.power(diff, 2)))

                truth_to_un_pcc_com = (np.corrcoef(sutru[:, 0], sun[:, 0])[0, 1] + np.corrcoef(sutru[:, 1], sun[:, 1])[0, 1] + np.corrcoef(sutru[:, 2], sun[:, 2])[0, 1]) / 3
                truth_to_est_pcc_com = (np.corrcoef(stru[:, 0], sest[:, 0])[0, 1] + np.corrcoef(stru[:, 1], sest[:, 1])[0, 1] + np.corrcoef(stru[:, 2], sest[:, 2])[0, 1]) / 3

                p = np.column_stack((np.arange(len(stru)), stru))
                q = np.column_stack((np.arange(len(sest)), sest))
                truth_to_est_fr_com = frechet_dist(p, q)
                truth_to_est_dtw_com = dtw.dtw(p, q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

                p = np.column_stack((np.arange(len(sutru)), sutru))
                q = np.column_stack((np.arange(len(sun)), sun))
                truth_to_un_fr_com = frechet_dist(p, q)
                truth_to_un_dtw_com = dtw.dtw(p, q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

                # XCOM
                est = data.down_kinect_xcom[:length][o:-o]
                un = data.down_kinect_unfiltered_xcom[:length][o:-o]
                tru = data.down_theia_xcom_15_hz[:length][o:-o]

                # sun, sutru = corr_shift_trim3d(un, tru)
                # sest, stru = corr_shift_trim3d(est, tru)

                sun, sutru = un, tru
                sest, stru = est, tru

                diff = np.linalg.norm(stru - sest, axis=1)
                truth_to_est_rmse_xcom = np.sqrt(np.mean(np.power(diff, 2)))

                diff = np.linalg.norm(sutru - sun, axis=1)
                truth_to_un_rmse_xcom = np.sqrt(np.mean(np.power(diff, 2)))

                truth_to_un_pcc_xcom = (np.corrcoef(sutru[:, 0], sun[:, 0])[0, 1] + np.corrcoef(sutru[:, 1], sun[:, 1])[0, 1] + np.corrcoef(sutru[:, 2], sun[:, 2])[0, 1]) / 3
                truth_to_est_pcc_xcom = (np.corrcoef(stru[:, 0], sest[:, 0])[0, 1] + np.corrcoef(stru[:, 1], sest[:, 1])[0, 1] + np.corrcoef(stru[:, 2], sest[:, 2])[0, 1]) / 3

                p = np.column_stack((np.arange(len(stru)), stru))
                q = np.column_stack((np.arange(len(sest)), sest))
                truth_to_est_fr_xcom = frechet_dist(p, q)
                truth_to_est_dtw_xcom = dtw.dtw(p, q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

                p = np.column_stack((np.arange(len(sutru)), sutru))
                q = np.column_stack((np.arange(len(sun)), sun))
                truth_to_un_fr_xcom = frechet_dist(p, q)
                truth_to_un_dtw_xcom = dtw.dtw(p, q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

                com_record = (
                    ex_name,
                    short_name(filter_name),
                    best_factor,
                    truth_to_est_rmse_com,
                    truth_to_est_dtw_com,
                    truth_to_est_fr_com,
                    truth_to_est_pcc_com,
                )
                com_records.append(com_record)

                xcom_record = (
                    ex_name,
                    short_name(filter_name),
                    best_factor,
                    truth_to_est_rmse_xcom,
                    truth_to_est_dtw_xcom,
                    truth_to_est_fr_xcom,
                    truth_to_est_pcc_xcom,
                )
                xcom_records.append(xcom_record)

                '''
                print(f"Filter: {filter_name}")
                print(f"mean_pred_to_est_rmse: {np.array(l_pred_to_est_rmse).mean()}")
                print(f"mean_truth_to_pred_rmse: {np.array(l_truth_to_pred_rmse).mean()}")
                print(f"mean_truth_to_est_rmse: {np.array(l_truth_to_est_rmse).mean()}")
                print(f"mean_truth_to_unfiltered_rmse: {np.array(l_truth_to_unfiltered_rmse).mean()}")

                print(f"max_pred_to_est_rmse: {np.array(l_pred_to_est_rmse).max()}")
                print(f"max_truth_to_pred_rmse: {np.array(l_truth_to_pred_rmse).max()}")
                print(f"max_truth_to_est_rmse: {np.array(l_truth_to_est_rmse).max()}")
                print(f"max_truth_to_unfiltered_rmse: {np.array(l_truth_to_unfiltered_rmse).max()}")

                print(f"min_pred_to_est_rmse: {np.array(l_pred_to_est_rmse).min()}")
                print(f"min_truth_to_pred_rmse: {np.array(l_truth_to_pred_rmse).min()}")
                print(f"min_truth_to_est_rmse: {np.array(l_truth_to_est_rmse).min()}")
                print(f"min_truth_to_unfiltered_rmse: {np.array(l_truth_to_unfiltered_rmse).min()}")

                print(f"std_pred_to_est_rmse: {np.array(l_pred_to_est_rmse).std()}")
                print(f"std_truth_to_pred_rmse: {np.array(l_truth_to_pred_rmse).std()}")
                print(f"std_truth_to_est_rmse: {np.array(l_truth_to_est_rmse).std()}")
                print(f"std_truth_to_unfiltered_rmse: {np.array(l_truth_to_unfiltered_rmse).std()}")

                print()
                print(f"mean_truth_to_est_fr: {np.array(l_truth_to_est_fr).mean()}")
                print(f"mean_truth_to_unfiltered_fr: {np.array(l_truth_to_un_fr).mean()}")
                print()

                print()
                print(f"mean_truth_to_est_dtw: {np.array(l_truth_to_est_dtw).mean()}")
                print(f"mean_truth_to_unfiltered_dtw: {np.array(l_truth_to_un_dtw).mean()}")
                print()

                print(f"Filter: {filter_name}")
                print(f"truth_to_est_rmse_com: {truth_to_est_rmse_com}")
                print(f"truth_to_un_rmse_com: {truth_to_un_rmse_com}")
                print()

                print(f"truth_to_est_rmse_xcom: {truth_to_est_rmse_xcom}")
                print(f"truth_to_un_rmse_xcom: {truth_to_un_rmse_xcom}")
                print()
                '''

        recorddata = np.array(records, dtype=[
            ("Experiment Name", f"U{len('s30001')}"),
            ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
            ("Joint Name", f"U{len('Right Shoulder')}"),
            ("Factor", "f"),
            ("RMSE", "f"),
            ("DTW", "f"),
            ("DFD", "f"),
            ("PCC", "f"),
        ])
        dataframe = pd.DataFrame.from_records(recorddata)

        com_recorddata = np.array(com_records, dtype=[
            ("Experiment Name", f"U{len('s30001')}"),
            ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
            ("Factor", "f"),
            ("RMSE", "f"),
            ("DTW", "f"),
            ("DFD", "f"),
            ("PCC", "f"),
        ])
        com_dataframe = pd.DataFrame.from_records(com_recorddata)

        xcom_recorddata = np.array(xcom_records, dtype=[
            ("Experiment Name", f"U{len('s30001')}"),
            ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
            ("Factor", "f"),
            ("RMSE", "f"),
            ("DTW", "f"),
            ("DFD", "f"),
            ("PCC", "f"),
        ])
        xcom_dataframe = pd.DataFrame.from_records(xcom_recorddata)

        oldsize = plt.rcParams['figure.figsize']
        plt.rcParams["figure.figsize"] = 40, 40
        sns.set_theme(rc={'figure.figsize':(40, 40)})
        for metric, ylabel in zip(METRIC_NAMES, YLABELS):
            for filter_name in FILTER_TYPES:
                sub = dataframe.loc[dataframe["Filter Name"] == short_name(filter_name)]
                subsub = sub.loc[sub["Experiment Name"] == ex_name]
                sns.set_style("darkgrid")
                if vel:
                    subsub=subsub.loc[subsub["Factor"] > 9]
                len_joints = len(MATCHING_JOINTS)
                markers=["^", "o", "X", "v", "s", "<", ">", "*", "D", "P", "^", "o", "X", "v", "s", "<", ">", "*", "D", "P"][:len_joints] + ["d"]
                linestyles=(["-"] * 8 + [":"] * 8)[:len_joints] + ["-."]
                ax = sns.catplot(
                    data=subsub,
                    x="Factor",
                    y=metric,
                    kind="point",
                    hue="Joint Name",
                    markers=markers,
                    linestyles=linestyles,
                    markersize=4,
                    linewidth=1,
                    legend_out=True,
                )
                ax.set_xticklabels(rotation=40, ha="right")
                ticks = ax.ax.get_xticklabels()
                plt.setp(ax.ax.get_xticklabels(), visible=False)
                plt.setp(ticks[::2], visible=True)
                plt.xlabel(r"$\lambda$")
                # Otherwise take default
                if ylabel:
                    plt.ylabel(ylabel)
                plt.title(rf"Ex: {map_ex_name(ex_name)} - {filter_name} {metric} for different $\lambda$ for each joint")
                os.makedirs(f"./results/experiments/{filter_name}/over_factor/{ex_name}/", exist_ok=True)
                if vel:
                    plt.savefig(f"./results/experiments/{filter_name}/over_factor/{ex_name}/{metric}_over_factor_vel.pdf", bbox_inches='tight')
                else:
                    plt.savefig(f"./results/experiments/{filter_name}/over_factor/{ex_name}/{metric}_over_factor.pdf", bbox_inches="tight")
                plt.cla()

                # Plot CoM
                sub = com_dataframe.loc[com_dataframe["Filter Name"] == short_name(filter_name)]
                subsub = sub.loc[sub["Experiment Name"] == ex_name]
                sns.set_style("darkgrid")
                if vel:
                    subsub=subsub.loc[subsub["Factor"] > 9]
                ax = sns.catplot(
                    data=subsub,
                    x="Factor",
                    y=metric,
                    kind="point",
                    markersize=4,
                    linewidth=1,
                    legend_out=False,
                )
                ax.set_xticklabels(rotation=40, ha="right")
                ticks = ax.ax.get_xticklabels()
                plt.setp(ax.ax.get_xticklabels(), visible=False)
                plt.setp(ticks[::2], visible=True)
                plt.xlabel(r"$\lambda$")
                # Otherwise take default
                if ylabel:
                    plt.ylabel(ylabel)
                plt.title(rf"Ex: {map_ex_name(ex_name)} - {filter_name} {metric} for CoM")
                os.makedirs(f"./results/experiments/{filter_name}/over_factor/{ex_name}/", exist_ok=True)
                if vel:
                    plt.savefig(f"./results/experiments/{filter_name}/over_factor/{ex_name}/{metric}_over_factor_vel_com.pdf", bbox_inches="tight")
                else:
                    plt.savefig(f"./results/experiments/{filter_name}/over_factor/{ex_name}/{metric}_over_factor_com.pdf", bbox_inches="tight")
                plt.cla()

                # Plot XcoM
                sub = xcom_dataframe.loc[xcom_dataframe["Filter Name"] == short_name(filter_name)]
                subsub = sub.loc[sub["Experiment Name"] == ex_name]
                sns.set_style("darkgrid")
                ax = sns.catplot(
                    data=subsub.loc[subsub["Factor"] > 9],
                    x="Factor",
                    y=metric,
                    kind="point",
                    markersize=4,
                    linewidth=1,
                    legend_out=False,
                )
                ax.set_xticklabels(rotation=40, ha="right")
                ticks = ax.ax.get_xticklabels()
                plt.setp(ax.ax.get_xticklabels(), visible=False)
                plt.setp(ticks[::2], visible=True)
                plt.xlabel(r"$\lambda$")
                # Otherwise take default
                if ylabel:
                    plt.ylabel(ylabel)
                plt.title(rf"Ex: {map_ex_name(ex_name)} - {filter_name} {metric} for XcoM")
                os.makedirs(f"./results/experiments/{filter_name}/over_factor/{ex_name}/", exist_ok=True)
                if vel:
                    plt.savefig(f"./results/experiments/{filter_name}/over_factor/{ex_name}/{metric}_over_factor_vel_xcom.pdf", bbox_inches="tight")
                else:
                    plt.savefig(f"./results/experiments/{filter_name}/over_factor/{ex_name}/{metric}_over_factor_xcom.pdf", bbox_inches="tight")
                plt.cla()
        # plt.rcParams["figure.figsize"] = oldsize

    recorddata = np.array(records, dtype=[
        ("Experiment Name", f"U{len('s30001')}"),
        ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
        ("Joint Name", f"U{len('Right Shoulder')}"),
        ("Factor", "f"),
        ("RMSE", "f"),
        ("DTW", "f"),
        ("DFD", "f"),
        ("PCC", "f"),
    ])
    dataframe = pd.DataFrame.from_records(recorddata)

    com_recorddata = np.array(com_records, dtype=[
        ("Experiment Name", f"U{len('s30001')}"),
        ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
        ("Factor", "f"),
        ("RMSE", "f"),
        ("DTW", "f"),
        ("DFD", "f"),
        ("PCC", "f"),
    ])
    com_dataframe = pd.DataFrame.from_records(com_recorddata)

    xcom_recorddata = np.array(xcom_records, dtype=[
        ("Experiment Name", f"U{len('s30001')}"),
        ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
        ("Factor", "f"),
        ("RMSE", "f"),
        ("DTW", "f"),
        ("DFD", "f"),
        ("PCC", "f"),
    ])
    xcom_dataframe = pd.DataFrame.from_records(xcom_recorddata)

    for metric, ylabel in zip(METRIC_NAMES, YLABELS):
        for filter_name in FILTER_TYPES:
            sub = dataframe.loc[dataframe["Filter Name"] == short_name(filter_name)]
            sns.set_style("darkgrid")
            len_joints = len(MATCHING_JOINTS)
            markers=["^", "o", "X", "v", "s", "<", ">", "*", "D", "P", "^", "o", "X", "v", "s", "<", ">", "*", "D", "P"][:len_joints] + ["d"]
            linestyles=(["-"] * 8 + [":"] * 8)[:len_joints] + ["-."]
            if vel:
                sub=sub.loc[sub["Factor"] > 9]
            ax = sns.catplot(
                data=sub,
                x="Factor",
                y=metric,
                kind="point",
                hue="Joint Name",
                markers=markers,
                linestyles=linestyles,
                markersize=4,
                linewidth=1,
                legend_out=True,
                errorbar=None
            )
            ax.set_xticklabels(rotation=40, ha="right")
            ticks = ax.ax.get_xticklabels()
            plt.setp(ax.ax.get_xticklabels(), visible=False)
            plt.setp(ticks[::2], visible=True)
            plt.xlabel(r"$\lambda$")
            # Otherwise take default
            if ylabel:
                plt.ylabel(ylabel)
            plt.title(rf"Ex: Mean 2a\&b - {filter_name} {metric} for different $\lambda$ for each joint")
            os.makedirs(f"./results/experiments/{filter_name}/over_factor/s3000x", exist_ok=True)
            if vel:
                plt.savefig(f"./results/experiments/{filter_name}/over_factor/s3000x/{metric}_over_factor_vel.pdf", bbox_inches="tight")
            else:
                plt.savefig(f"./results/experiments/{filter_name}/over_factor/s3000x/{metric}_over_factor.pdf", bbox_inches="tight")
            plt.cla()

            # com
            sub = com_dataframe.loc[com_dataframe["Filter Name"] == short_name(filter_name)]
            if vel:
                sub=sub.loc[sub["Factor"] > 9]
            sns.set_style("darkgrid")
            ax = sns.catplot(
                data=sub,
                x="Factor",
                y=metric,
                kind="point",
                markersize=4,
                linewidth=1,
                legend_out=False,
                errorbar=None
            )
            ax.set_xticklabels(rotation=40, ha="right")
            ticks = ax.ax.get_xticklabels()
            plt.setp(ax.ax.get_xticklabels(), visible=False)
            plt.setp(ticks[::2], visible=True)
            plt.xlabel(r"$\lambda$")
            # Otherwise take default
            if ylabel:
                plt.ylabel(ylabel)
            plt.title(rf"Ex: Mean 2a\&b - {filter_name} {metric} for CoM")
            os.makedirs(f"./results/experiments/{filter_name}/over_factor/s3000x", exist_ok=True)
            if vel:
                plt.savefig(f"./results/experiments/{filter_name}/over_factor/s3000x/{metric}_over_factor_vel_com.pdf", bbox_inches="tight")
            else:
                plt.savefig(f"./results/experiments/{filter_name}/over_factor/s3000x/{metric}_over_factor_com.pdf", bbox_inches="tight")
            plt.cla()

            # xcom
            sub = xcom_dataframe.loc[xcom_dataframe["Filter Name"] == short_name(filter_name)]
            sns.set_style("darkgrid")
            ax = sns.catplot(
                data=sub,
                x="Factor",
                y=metric,
                kind="point",
                markersize=4,
                linewidth=1,
                legend_out=False,
                errorbar=None
            )
            ax.set_xticklabels(rotation=40, ha="right")
            ticks = ax.ax.get_xticklabels()
            plt.setp(ax.ax.get_xticklabels(), visible=False)
            plt.setp(ticks[::2], visible=True)
            plt.xlabel(r"$\lambda$")
            # Otherwise take default
            if ylabel:
                plt.ylabel(ylabel)
            plt.title(rf"Ex: Mean 2a\&b - {filter_name} {metric} for XcoM")
            os.makedirs(f"./results/experiments/{filter_name}/over_factor/s3000x", exist_ok=True)
            if vel:
                plt.savefig(f"./results/experiments/{filter_name}/over_factor/s3000x/{metric}_over_factor_vel_xcom.pdf", bbox_inches="tight")
            else:
                plt.savefig(f"./results/experiments/{filter_name}/over_factor/s3000x/{metric}_over_factor_xcom.pdf", bbox_inches="tight")
            plt.cla()

        sub = dataframe.loc[dataframe["Joint Name"] == "Mean"]
        if vel:
            sub=sub.loc[sub["Factor"] > 9]
        sns.set_style("darkgrid")
        len_joints = len(MATCHING_JOINTS)
        markers=["^", "o", "X", "v", "s", "<", ">", "*", "D", "P", "^", "o", "X", "v", "s", "<", ">", "*", "D", "P"][:len_joints] + ["d"]
        linestyles=(["-"] * 8 + [":"] * 8)[:len_joints] + ["-."]
        ax = sns.catplot(
            data=sub,
            x="Factor",
            y=metric,
            kind="point",
            hue="Filter Name",
            markersize=0,
            linewidth=1,
            legend_out=False,
            errorbar=None
        )
        ax.set_xticklabels(rotation=40, ha="right")
        ticks = ax.ax.get_xticklabels()
        plt.setp(ax.ax.get_xticklabels(), visible=False)
        plt.setp(ticks[::2], visible=True)
        plt.xlabel(r"$\lambda$")
        # Otherwise take default
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(rf"Ex: Mean 2a\&b - All Filters {metric} for different $\lambda$ for each joint")
        os.makedirs(f"./results/experiments/over_factor/s3000x", exist_ok=True)
        if vel:
            plt.savefig(f"./results/experiments/over_factor/s3000x/{metric}_over_factor_vel.pdf", bbox_inches="tight")
        else:
            plt.savefig(f"./results/experiments/over_factor/s3000x/{metric}_over_factor.pdf", bbox_inches="tight")
        plt.cla()

        # com
        if vel:
            com_dataframe = com_dataframe.loc[com_dataframe["Factor"] > 9]
        sns.set_style("darkgrid")
        ax = sns.catplot(
            data=com_dataframe,
            x="Factor",
            y=metric,
            kind="point",
            hue="Filter Name",
            markersize=0,
            linewidth=1,
            legend_out=False,
            errorbar=None
        )
        ax.set_xticklabels(rotation=40, ha="right")
        ticks = ax.ax.get_xticklabels()
        plt.setp(ax.ax.get_xticklabels(), visible=False)
        plt.setp(ticks[::2], visible=True)
        plt.xlabel(r"$\lambda$")
        # Otherwise take default
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(rf"Ex: Mean 2a\&b - All Filters {metric} for CoM")
        os.makedirs(f"./results/experiments/over_factor/s3000x", exist_ok=True)
        if vel:
            plt.savefig(f"./results/experiments/over_factor/s3000x/{metric}_over_factor_vel_com.pdf", bbox_inches="tight")
        else:
            plt.savefig(f"./results/experiments/over_factor/s3000x/{metric}_over_factor_com.pdf", bbox_inches="tight")
        plt.cla()

        # xcom
        sns.set_style("darkgrid")
        ax = sns.catplot(
            data=xcom_dataframe.loc[xcom_dataframe["Factor"] > 9],
            x="Factor",
            y=metric,
            kind="point",
            hue="Filter Name",
            markersize=0,
            linewidth=1,
            legend_out=False,
            errorbar=None
        )
        ax.set_xticklabels(rotation=40, ha="right")
        ticks = ax.ax.get_xticklabels()
        plt.setp(ax.ax.get_xticklabels(), visible=False)
        plt.setp(ticks[::2], visible=True)
        plt.xlabel(r"$\lambda$")
        # Otherwise take default
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(rf"Ex: Mean 2a\&b - All Filters {metric} for XcoM")
        os.makedirs(f"./results/experiments/over_factor/s3000x", exist_ok=True)
        if vel:
            plt.savefig(f"./results/experiments/over_factor/s3000x/{metric}_over_factor_vel_xcom.pdf", bbox_inches="tight")
        else:
            plt.savefig(f"./results/experiments/over_factor/s3000x/{metric}_over_factor_xcom.pdf", bbox_inches="tight")
        plt.cla()

def compare_prediction_vs_truth_for_different_filters_qtm_for_com(experiment_path: Path, ex_name: str, cutoff: float = 0.1, vel: bool = False) -> None:
    cop_records = []
    for ex_name in tqdm([f"s1000{i}" for i in range(1, 5)]):
        for best_factor in tqdm(np.arange(0, 150, 5)):
            for filter_name in ["ConstrainedSkeletonFilter", "SkeletonFilter", "SimpleConstrainedSkeletonFilter", "SimpleSkeletonFilter"]:
                path = experiment_path / filter_name / ex_name
                data = load_processed_data(find_factor_path(best_factor, path))


                length = min(data.down_kinect_com.shape[0], data.down_kinect_unfiltered_com.shape[0], data.qtm_cop.shape[0])
                o = max(int(length * cutoff), 1)

                est = data.down_kinect_com[:length][o:-o][:, :2]
                un = data.down_kinect_unfiltered_com[:length][o:-o][:, :2]
                tru = downsample(data.qtm_cop, data.qtm_cop_ts, 15)[:, :2][:length][o:-o]

                # sun, sutru = corr_shift_trim3d(un, tru, idx=0)
                # sest, stru = corr_shift_trim3d(est, tru, idx=0)

                sun, sutru = un, tru
                sest, stru = est, tru


                diff = np.linalg.norm(stru - sest, axis=1)
                truth_to_est_rmse_cop = np.sqrt(np.mean(np.power(diff, 2)))

                diff = np.linalg.norm(sutru - sun, axis=1)
                truth_to_un_cop = np.sqrt(np.mean(np.power(diff, 2)))

                truth_to_un_pcc_cop = (np.corrcoef(sutru[:, 0], sun[:, 0])[0, 1] + np.corrcoef(sutru[:, 1], sun[:, 1])[0, 1]) / 2
                truth_to_est_pcc_cop = (np.corrcoef(stru[:, 0], sest[:, 0])[0, 1] + np.corrcoef(stru[:, 1], sest[:, 1])[0, 1]) / 2

                p = np.column_stack((np.arange(len(stru)), stru))
                q = np.column_stack((np.arange(len(sest)), sest))
                truth_to_est_fr_cop = frechet_dist(p, q)
                truth_to_est_dtw_cop = dtw.dtw(p, q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

                p = np.column_stack((np.arange(len(sutru)), sutru))
                q = np.column_stack((np.arange(len(sun)), sun))
                truth_to_un_fr_cop = frechet_dist(p, q)
                truth_to_un_dtw_cop = dtw.dtw(p, q, step_pattern=dtw.rabinerJuangStepPattern(6, "c")).distance

                cop_record = (
                    ex_name,
                    short_name(filter_name),
                    best_factor,
                    truth_to_est_rmse_cop,
                    truth_to_est_dtw_cop,
                    truth_to_est_fr_cop,
                    truth_to_est_pcc_cop,
                )
                cop_records.append(cop_record)

    cop_recorddata = np.array(cop_records, dtype=[
        ("Experiment Name", f"U{len('s30001')}"),
        ("Filter Name", f"U{len('SimpleConstrainedSkeletonFilter')}"),
        ("Factor", "f"),
        ("RMSE", "f"),
        ("DTW", "f"),
        ("DFD", "f"),
        ("PCC", "f"),
    ])
    cop_dataframe = pd.DataFrame.from_records(cop_recorddata)

    plt.rcParams["figure.figsize"] = 40, 40
    sns.set_theme(rc={'figure.figsize':(40, 40)})
    for metric, ylabel in zip(METRIC_NAMES, YLABELS):
        # Plot CoP
        sns.set_style("darkgrid")
        ax = sns.catplot(
            data=cop_dataframe,
            x="Factor",
            y=metric,
            hue="Filter Name",
            kind="point",
            markersize=0,
            linewidth=1,
            legend_out=False,
            errorbar=None,
        )
        ax.set_xticklabels(rotation=40, ha="right")
        ticks = ax.ax.get_xticklabels()
        plt.setp(ax.ax.get_xticklabels(), visible=False)
        plt.setp(ticks[::2], visible=True)
        plt.xlabel(r"$\lambda$")
        # Otherwise take default
        if ylabel:
            plt.ylabel(ylabel)
        plt.title(rf"Ex: Mean 1a\&b - All Filters {metric} for CoM/CoP")
        os.makedirs(f"./results/experiments/over_factor/s1000x/", exist_ok=True)
        if vel:
            plt.savefig(f"./results/experiments/over_factor/s1000x/{metric}_over_factor_vel_cop.pdf", bbox_inches="tight")
        else:
            plt.savefig(f"./results/experiments/over_factor/s1000x/{metric}_over_factor_cop.pdf", bbox_inches="tight")
        plt.cla()



def create_joint_length_plots_and_table(path: Path):
    records = []
    focus_ex_names = ["s10001", "s30001"]
    factors = [5, 10, 35, 70]
    for factor in factors:
        for filter_type in FILTER_TYPES:
            for ex_name in EX_NAMES:
                ex_path = path / filter_type / ex_name
                data = cond_load_data(find_factor_path(factor, ex_path))
                global FILTER_NAME
                FILTER_NAME = filter_type
                recs = plot_constrained_segment_joint_length_change(ex_name, data, cutoff=0.1)
                records.extend(recs)

    recorddata = np.array(records, dtype=[
        ("Experiment Name", f"U{len('s30001')}"),
        ("Filter Type", f"U{len('SimpleConstrainedSkeletonFilter')}"),
        ("Factor", "f"),
        ("Segment Name", f"U{len('Upper Arm Right')}"),
        (r"filteredmean", "f"),
        (r"filteredvar", "f"),
        (r"unfilteredmean", "f"),
        (r"unfilteredvar", "f"),
        (r"theiamean", "f"),
        (r"theiavar", "f"),
    ])

    # header_names = [r"Experiment\\Name",r"Filter\\Type", r"Segment\\Name", "Factor", r"Filtered Var", r"Unfiltered Var", r"Theia Var"]
    header_names = [r"Experiment\\Name",r"Filter\\Type", r"Segment Name", r"$\lambda$", r"Filtered Var", r"Unfiltered Var"]
    some = [("UP_LEFT", ("Upper Arm Left", "Lower Arm Left")), ("DOWN_RIGHT", ("Thigh Right", "Shank Right"))]
    for key, value in some:
        a_name, b_name = value
        whole_segment_name = rf"{a_name} \& {b_name} Length Change"
        ref = f"{key}-length-change"
        table = GenericTableGeneration(header_names, whole_segment_name, ref)
        for ex_name in focus_ex_names:
            dataframe = pd.DataFrame.from_records(recorddata)
            sub = dataframe.loc[dataframe["Experiment Name"] == ex_name]
            for factor in [5, 35]:
                for filter_type in ["SkeletonFilter", "ConstrainedSkeletonFilter"]:
                    subsub = sub.loc[sub["Filter Type"] == filter_type]
                    sss = subsub.loc[subsub["Segment Name"] == a_name]
                    ssss = sss.loc[sss["Factor"] == factor]
                    row = (
                        map_ex_name(ex_name),
                        rf"\{short_name(filter_type)}",
                        a_name,
                        str(factor),
                        rf"\SI{{{round(ssss['filteredvar'].values[0], 9):.5e}}}{{\meter}}",
                        rf"\SI{{{round(ssss['unfilteredvar'].values[0], 9):.5e}}}{{\meter}}",
                        # rf"${round(ssss['theiavar'].values[0], 9):.5e}$ m",
                    )
                    table.append(row)
                    sss = subsub.loc[subsub["Segment Name"] == b_name]
                    ssss = sss.loc[sss["Factor"] == factor]

                    row = (
                        map_ex_name(ex_name),
                        rf"\{short_name(filter_type)}",
                        b_name,
                        str(factor),
                        rf"\SI{{{round(ssss['filteredvar'].values[0], 9):.5e}}}{{\meter}}",
                        rf"\SI{{{round(ssss['unfilteredvar'].values[0], 9):.5e}}}{{\meter}}",
                        # rf"${round(ssss['theiavar'].values[0], 9):.5e}$ m",
                    )
                    table.append(row)

        end_path = f"{a_name.replace(' ', '-')}_{b_name.replace(' ', '-')}"
        out_path = Path(f"./results/experiments/joint_segment_lengths/{end_path}.tex")
        table.generate_table(out_path)

    baseline = dataframe.groupby(["Filter Type", "Factor"]).mean("unfilteredvar")["unfilteredvar"].values[0]

    sub = dataframe.groupby(["Filter Type", "Factor"])["filteredvar"].mean()

    markers=["^", "o", "X", "v", "s", "<", ">", "*", "D", "P", "^", "o", "X", "v", "s", "<", ">", "*", "D", "P"] + ["d"]
    sns.set_style("darkgrid")
    ax = sns.lineplot(
        data=sub.to_frame(),
        x="Factor",
        y="filteredvar",
        hue="Filter Type",
        markers=markers,
        markersize=4,
        linewidth=1,
    )
    ax.axhline(baseline, xmin=0, xmax=150, label="Unfiltered Variance", linestyle="dotted", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    # ax.set_xticklabels(rotation=40, ha="right")
    # ticks = ax.ax.get_xticklabels()
    # plt.setp(ax.ax.get_xticklabels(), visible=False)
    # plt.setp(ticks[::6], visible=True)
    plt.xlabel(r"$\lambda$")
    # Otherwise take default
    plt.ylabel(rf"Variance [m^{2}]")
    plt.legend()
    plt.title("Mean Length Change Variance over All Experiments")
    sns.set_style("darkgrid")
    os.makedirs(f"./results/experiments/joint_segment_lengths/sx000x", exist_ok=True)
    plt.savefig(f"./results/experiments/joint_segment_lengths/sx000x/mean_var.pdf", bbox_inches="tight")
    plt.cla()


    sub = dataframe.loc[dataframe["Filter Type"] == "SkeletonFilter"]
    subsub = sub.groupby(["Segment Name", "Factor"])["filteredvar"].mean()

    markers=["^", "o", "X", "v", "s", "<", ">", "*", "D", "P", "^", "o", "X", "v", "s", "<", ">", "*", "D", "P"] + ["d"]
    sns.set_style("darkgrid")
    ax = sns.lineplot(
        data=subsub.to_frame(),
        x="Factor",
        y="filteredvar",
        hue="Segment Name",
        markers=markers,
        markersize=4,
        linewidth=1,
    )
    ax.axhline(baseline, xmin=0, xmax=150, label="Unfiltered Var", linestyle="dotted", alpha=0.6)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    # ax.set_xticklabels(rotation=40, ha="right")
    # ticks = ax.ax.get_xticklabels()
    # plt.setp(ax.ax.get_xticklabels(), visible=False)
    # plt.setp(ticks[::6], visible=True)
    plt.xlabel(r"$\lambda$")
    # Otherwise take default
    plt.ylabel(rf"Variance [m^{2}]")
    plt.title("Mean Length Change Variance for SkeletonFilter for each Segment over all Experiments")
    sns.set_style("darkgrid")
    os.makedirs(f"./results/experiments/joint_segment_lengths/sx000x", exist_ok=True)
    plt.savefig(f"./results/experiments/joint_segment_lengths/sx000x/mean_var_segments.pdf", bbox_inches="tight")
    plt.cla()


def compare_frames(path: Path) -> None:

    d = {"Type": [], "FPS": [],  "Percentage": [], "Experiment Name": []}
    total_dataframe = pd.DataFrame(data=d)
    for ex_name in THEIA_EX_NAMES:
        fast_name = ex_name
        slow_name = "b_"  + ex_name

        p = path / "SimpleSkeletonFilter"

        fast = cond_load_data(find_factor_path(0, p / fast_name))
        slow = cond_load_data(find_factor_path(0, p / slow_name))

        ts_fast = 1./ (fast.kinect_ts[1:] - fast.kinect_ts[:-1])
        ts_slow = 1./ (slow.kinect_ts[1:] - slow.kinect_ts[:-1])

        f = np.round(ts_fast)
        s = np.round(ts_slow)

        bin_f, count_f = np.unique(f, return_counts=True)
        bin_s, count_s = np.unique(s, return_counts=True)

        count_f = np.divide(count_f, len(ts_fast)) * 100
        count_s = np.divide(count_s, len(ts_fast)) * 100

        normed_f = np.zeros_like(bin_s)
        lf = len(bin_f)
        ls = len(bin_s)
        normed_f[ls - lf:] = count_f

        d = {"Type": ["Lean NN"] * ls + ["Normal NN"] * ls, "FPS": np.hstack((bin_s, bin_s)), "Percentage": np.hstack((normed_f, count_s)), "Experiment Name": [fast_name] * 2 * ls}

        data = pd.DataFrame(data=d)
        total_dataframe = pd.concat([total_dataframe, data])

    sns.set_style("darkgrid")
    ax = sns.barplot(
        x="Percentage",
        y="FPS",
        hue="Type",
        data=total_dataframe,
        orient='h'
    )
    plt.xscale("log")
    plt.xlim(right=130)
    '''
    ax = plt.gca()
    y_max = data['Percentage'].value_counts().max()
    ax.set_ylim(1)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.2f}'.format(p.get_height()),
            fontsize=12, color='red', ha='center', va='bottom')
    '''
    for container in ax.containers:
        ax.bar_label(container, fmt=lambda value: rf"{value:.3f}\%" if value != 0 else "", padding=25)
        # ax.bar_label(container, fmt='%.2f\%')
    plt.legend()
    plt.savefig(f"./results/experiments/benchmark/fps-mean.pdf", bbox_inches="tight")
    plt.cla()


if __name__ == "__main__":
    main()
