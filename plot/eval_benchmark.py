from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
from pprint import pprint as pp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import seaborn as sns

NAME_MAP = {
    "apply_filter": "Apply Filter",
    "build_filter": "Build Filter",
    "camera": "Camera",
    "detect_floor": "Detect Floor",
    "extract_stability_metrics": "Extract Stability Metric",
    "imu": "Extract IMU Data",
    "measurement_queue_produce": "Produce onto Measurement Queue",
    "network": "Network Inference",
    "pointcloud": "Generate Pointcould",
    "process_queue_produce": "Produce onto Process Queue",
    "recording_body": "Save Body to Recording",
    "recording_imu": "Save IMU to Recording",
    "save_body": "Save Body Information",
    "save_imu": "Save IMU Inforamtion",
    "step": "Perform Filter Step",
    "visualize": "Realtime Data Visualization",
    "rest": "Later Sections"
}

@dataclass
class Metric:
    name: str | None
    min: float | None
    max: float | None
    mean: float | None
    var: float | None

    def zero_to_null(self) -> None:
        if self.min == 0:
            self.min = None
        if self.max == 0:
            self.max = None
        if self.mean == 0:
            self.mean = None
        if self.var == 0:
            self.var = None

    def __add__(self, other) -> Metric:
        self.add_value(other)
        return self

    def add_value(self, other_metric: Metric) -> None:
        if self.min is not None and other_metric.min is not None:
            self.min += other_metric.min;
        if self.max is not None and other_metric.max is not None:
            self.max += other_metric.max;
        if self.mean is not None and other_metric.mean is not None:
            self.mean += other_metric.mean;
        if self.var is not None and other_metric.var is not None:
            self.var += other_metric.var;

    def divide_by(self, length: float) -> None:
        """Helper method to convert zero value to None.

        Used for mean data where we start with 0 and never add for only none
        values. Converts the 0 back to None, if any number ever has non None we
        expect to not have 0 at the end.
        """
        if self.min:
            self.min /= length;
        if self.max:
            self.max /= length;
        if self.mean:
            self.mean /= length;
        if self.var:
            self.var /= length;

@dataclass
class BenchData:
    apply_filter: Metric
    build_filter: Metric
    camera: Metric
    detect_floor: Metric
    extract_stability_metrics: Metric
    imu: Metric
    measurement_queue_produce: Metric
    network: Metric
    pointcloud: Metric
    process_queue_produce: Metric
    recording_body: Metric
    recording_imu: Metric
    save_body: Metric
    save_imu: Metric
    step: Metric
    visualize: Metric

    def sub_ordered(self) -> list[Metric]:
        return [
            self.imu,
            self.save_body,
            self.save_imu,
            self.pointcloud,
            self.detect_floor,
            self.apply_filter,
            self.build_filter,
            self.step,
            self.extract_stability_metrics,
            self.visualize,
        ]

    def rest(self) -> list[Metric]:
        rest = self.imu+ self.save_body+ self.save_imu+ self.pointcloud+ self.measurement_queue_produce+ self.detect_floor+ self.apply_filter+ self.build_filter+ self.step+ self.extract_stability_metrics+ self.process_queue_produce+ self.visualize
        rest.name = "rest"
        return [
            self.network,
            rest
        ]

    def ordered(self) -> list[Metric]:
        return [
            self.camera,
            self.recording_body,
            self.recording_imu,
            self.network,
            self.imu,
            self.save_body,
            self.save_imu,
            self.pointcloud,
            self.measurement_queue_produce,
            self.detect_floor,
            self.apply_filter,
            self.build_filter,
            self.step,
            self.extract_stability_metrics,
            self.process_queue_produce,
            self.visualize,
        ]

    def add_value(self, other_bench_data: BenchData) -> None:
        self.apply_filter.add_value(other_bench_data.apply_filter)
        self.build_filter.add_value(other_bench_data.build_filter)
        self.camera.add_value(other_bench_data.camera)
        self.detect_floor.add_value(other_bench_data.detect_floor)
        self.extract_stability_metrics.add_value(other_bench_data.extract_stability_metrics)
        self.imu.add_value(other_bench_data.imu)
        self.measurement_queue_produce.add_value(other_bench_data.measurement_queue_produce)
        self.network.add_value(other_bench_data.network)
        self.pointcloud.add_value(other_bench_data.pointcloud)
        self.process_queue_produce.add_value(other_bench_data.process_queue_produce)
        self.recording_body.add_value(other_bench_data.recording_body)
        self.recording_imu.add_value(other_bench_data.recording_imu)
        self.save_body.add_value(other_bench_data.save_body)
        self.save_imu.add_value(other_bench_data.save_imu)
        self.step.add_value(other_bench_data.step)
        self.visualize.add_value(other_bench_data.visualize)

    def divide_by(self, length: float) -> None:
        self.apply_filter.divide_by(length)
        self.build_filter.divide_by(length)
        self.camera.divide_by(length)
        self.detect_floor.divide_by(length)
        self.extract_stability_metrics.divide_by(length)
        self.imu.divide_by(length)
        self.measurement_queue_produce.divide_by(length)
        self.network.divide_by(length)
        self.pointcloud.divide_by(length)
        self.process_queue_produce.divide_by(length)
        self.recording_body.divide_by(length)
        self.recording_imu.divide_by(length)
        self.save_body.divide_by(length)
        self.save_imu.divide_by(length)
        self.step.divide_by(length)
        self.visualize.divide_by(length)

    def zero_to_null(self) -> None:
        """Helper method to convert zero value to None.

        Used for mean data where we start with 0 and never add for only none
        values. Converts the 0 back to None, if any number ever has non None we
        expect to not have 0 at the end.
        """
        self.apply_filter.zero_to_null()
        self.build_filter.zero_to_null()
        self.camera.zero_to_null()
        self.detect_floor.zero_to_null()
        self.extract_stability_metrics.zero_to_null()
        self.imu.zero_to_null()
        self.measurement_queue_produce.zero_to_null()
        self.network.zero_to_null()
        self.pointcloud.zero_to_null()
        self.process_queue_produce.zero_to_null()
        self.recording_body.zero_to_null()
        self.recording_imu.zero_to_null()
        self.save_body.zero_to_null()
        self.save_imu.zero_to_null()
        self.step.zero_to_null()
        self.visualize.zero_to_null()

def empty_bench_data():
    """Return empty bench data."""
    return BenchData(
        apply_filter=Metric("apply_filter", 0, 0, 0, 0),
        build_filter=Metric("build_filter", 0, 0, 0, 0),
        camera=Metric("camera", 0, 0, 0, 0),
        detect_floor=Metric("detect_floor", 0, 0, 0, 0),
        extract_stability_metrics=Metric("extract_stability_metrics", 0, 0, 0, 0),
        imu=Metric("imu", 0, 0, 0, 0),
        measurement_queue_produce=Metric("measurement_queue_produce", 0, 0, 0, 0),
        network=Metric("network", 0, 0, 0, 0),
        pointcloud=Metric("pointcloud", 0, 0, 0, 0),
        process_queue_produce=Metric("process_queue_produce", 0, 0, 0, 0),
        recording_body=Metric("recording_body", 0, 0, 0, 0),
        recording_imu=Metric("recording_imu", 0, 0, 0, 0),
        save_body=Metric("save_body", 0, 0, 0, 0),
        save_imu=Metric("save_imu", 0, 0, 0, 0),
        step=Metric("step", 0, 0, 0, 0),
        visualize=Metric("visualize", 0, 0, 0, 0),
    )

def load_bench_data(path: Path) -> BenchData:
    """Load bench data."""
    with path.open(mode="r", encoding="UTF-8") as _json:
        bench = json.load(_json)
        print(path)
        return BenchData(
            apply_filter=Metric("apply_filter", bench["min"]["apply_filter"], bench["max"]["apply_filter"], bench["mean"]["apply_filter"], bench["var"]["apply_filter"]),
            build_filter=Metric("build_filter", bench["min"]["build_filter"], bench["max"]["build_filter"], bench["mean"]["build_filter"], bench["var"]["build_filter"]),
            camera=Metric("camera", bench["min"]["camera"], bench["max"]["camera"], bench["mean"]["camera"], bench["var"]["camera"]),
            detect_floor=Metric("detect_floor", bench["min"]["detect_floor"], bench["max"]["detect_floor"], bench["mean"]["detect_floor"], bench["var"]["detect_floor"]),
            extract_stability_metrics=Metric("extract_stability_metrics", bench["min"]["extract_stability_metrics"], bench["max"]["extract_stability_metrics"], bench["mean"]["extract_stability_metrics"], bench["var"]["extract_stability_metrics"]),
            imu=Metric("imu", bench["min"]["imu"], bench["max"]["imu"], bench["mean"]["imu"], bench["var"]["imu"]),
            measurement_queue_produce=Metric("measurement_queue_produce", bench["min"]["measurement_queue_produce"], bench["max"]["measurement_queue_produce"], bench["mean"]["measurement_queue_produce"], bench["var"]["measurement_queue_produce"]),
            network=Metric("network", bench["min"]["network"], bench["max"]["network"], bench["mean"]["network"], bench["var"]["network"]),
            pointcloud=Metric("pointcloud", bench["min"]["pointcloud"], bench["max"]["pointcloud"], bench["mean"]["pointcloud"], bench["var"]["pointcloud"]),
            process_queue_produce=Metric("process_queue_produce", bench["min"]["process_queue_produce"], bench["max"]["process_queue_produce"], bench["mean"]["process_queue_produce"], bench["var"]["process_queue_produce"]),
            recording_body=Metric("recording_body", bench["min"]["recording_body"], bench["max"]["recording_body"], bench["mean"]["recording_body"], bench["var"]["recording_body"]),
            recording_imu=Metric("recording_imu", bench["min"]["recording_imu"], bench["max"]["recording_imu"], bench["mean"]["recording_imu"], bench["var"]["recording_imu"]),
            save_body=Metric("save_body", bench["min"]["save_body"], bench["max"]["save_body"], bench["mean"]["save_body"], bench["var"]["save_body"]),
            save_imu=Metric("save_imu", bench["min"]["save_imu"], bench["max"]["save_imu"], bench["mean"]["save_imu"], bench["var"]["save_imu"]),
            step=Metric("step", bench["min"]["step"], bench["max"]["step"], bench["mean"]["step"], bench["var"]["step"]),
            visualize=Metric("visualize", bench["min"]["visualize"], bench["max"]["visualize"], bench["mean"]["visualize"], bench["var"]["visualize"]),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_json_dir")

    args = parser.parse_args()

    benchs = []
    for bench_json in Path(args.bench_json_dir).glob("*.json"):
        bench_data = load_bench_data(bench_json)
        benchs.append(bench_data)

    mean: BenchData = empty_bench_data()

    for bench in benchs:
        mean.add_value(bench)

    mean.divide_by(len(benchs))
    mean.zero_to_null()
    pp(mean)

    x = 1
    width = 0.5
    total_time = 0
    sns.set_style("darkgrid")
    for idx, top in enumerate(mean.sub_ordered()):
        if top.mean is not None:
            container = plt.barh(x * idx, top.mean, width, label=NAME_MAP[top.name], left=total_time)
            plt.errorbar(total_time + top.mean, x*idx, 0, np.sqrt(top.var), barsabove=True)
            plt.bar_label(container, labels=[f"{top.mean:.3f} ms ± {np.sqrt(top.var):.3f} ms"], padding=min(np.sqrt(top.var) * 45, 100), label_type="edge")
            total_time += top.mean



    names = ([NAME_MAP[element.name] for element in mean.sub_ordered()])
    plt.yticks(np.arange(len(names)), names)
    plt.xlabel("Time [milliseconds]")
    plt.ylabel("Steps")
    # plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    tdir = Path("results/experiments/benchmark")
    tdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(tdir / "benchmark.pdf", bbox_inches="tight")
    plt.cla()

    x = 1
    width = 0.5
    total_time = 0
    sns.set_style("darkgrid")
    for idx, top in enumerate(mean.rest()):
        if top.mean is not None:
            container = plt.barh(x * idx, top.mean, width, label=NAME_MAP[top.name], left=total_time)
            plt.errorbar(total_time + top.mean, x*idx, 0, np.sqrt(top.var), barsabove=True)
            plt.bar_label(container, labels=[f"{top.mean:.3f} ms ± {np.sqrt(top.var):.3f} ms"], padding=min(np.sqrt(top.var) * 15, 50), label_type="edge")
            total_time += top.mean



    names = ([NAME_MAP[element.name] for element in mean.rest()])
    plt.yticks(np.arange(len(names)), names)
    plt.xlabel("Time [seconds]")
    plt.ylabel("Steps")
    # plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    tdir = Path("results/experiments/benchmark")
    tdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(tdir / "benchmark-rest.pdf", bbox_inches="tight")
    plt.cla()

if __name__ == "__main__":
    main()
