#!/usr/bin/env bash
mkdir -p /tmp/results
for i in $(python -c 'import numpy as np; [print(i) for i in np.arange(5, 11, 1)]'); do
    cd /home/orb/repos/kinect_mocap_studio/
    JSON=$(./build/kinect_mocap_studio -i output_NFOV_UNBINNED_OFF_30fps_122.mkv | rg 'Frames written' | awk '{print$5}')
    echo "{\"qtm_file\": \"/home/orb/Desktop/Azure_Kinect_Marcus_Filipp_Edited/S20003.tsv\", \"kinect_file\": \"$JSON\"}" > ./experiments/S20003.json
    ./build/evaluate -e ./experiments/S20003.json -r 0 -p 0
    cd /home/orb/repos/master/code/plot
    python experiment_evaluate.py /home/orb/repos/kinect_mocap_studio/experiment_result/experiments/S20003 > /tmp/results/${i}.json
done
