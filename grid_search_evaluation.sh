#!/usr/bin/env bash
mkdir -p ./plot/results/simulation
for i in $(python -c 'import numpy as np; [print(i) for i in np.arange(0, 0.0016, 0.0001)]'); do
    ./build/load $i
    cd plot
    python simulation_evaluate.py > results/simulation/${i}.log
    cd ..
done
