#!/bin/bash

# path to python
python="/usr/local/bin/python3"

# path to visualizer.py
visualizer="/Users/megamot/Programming/Python/Phase-problem-modeling/executable/visualizer.py"

# path to true phase
filename='phi z=0.000.npy'
true_phase_file_path="/Users/megamot/Programming/Python/Phase-problem-modeling/data/executable_synthesis/$filename"

# saving params
save_folder="/Users/megamot/Programming/Python/Phase-problem-modeling/data/executable_views"

mode="ERROR"

dpi=100
figsize_width=12.4
figsize_height=7.8
px_size=5e-6
cmap="jet"
phase_ylabel="rad"
show_plot=0
save_plot=1

if [ "$1" != "" ]
then
  echo "$1"
  $python $visualizer --file_path "$1" --true_phase_file_path "$true_phase_file_path" --save_folder $save_folder --mode $mode --dpi $dpi --figsize $figsize_width $figsize_height --px_size $px_size --cmap $cmap --show_plot $show_plot --save_plot $save_plot --phase_ylabel $phase_ylabel
fi

# pause
