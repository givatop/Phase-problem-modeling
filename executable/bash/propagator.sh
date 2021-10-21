#!/bin/bash

RANDOM=$$

# wave params
wavelength=555e-9
px_size=5e-6
wave_path=$1

# propagation params
start=-0.000001
stop=0.000001
step=1e-6
method="angular_spectrum_band_limited"

# saving params
save_folder="/Users/m.konoplyov/Programming/Python/Phase-problem-modeling/data/executable_props/$RANDOM"
separate_save=1

# path to python
#python="/usr/local/bin/python3"

# path to propagator.py
propagator="/Users/megamot/Programming/Python/Phase-problem-modeling/executable/propagator.py"

echo running...
python3 $propagator --wavelength $wavelength  --px_size $px_size --wave_path "$wave_path" --start $start --stop $stop --step $step --method $method --save_folder "$save_folder" --separate_save $separate_save

# pause
