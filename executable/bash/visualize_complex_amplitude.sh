#!/bin/bash

# path to python
python="/usr/local/bin/python3"

# path to visualizer.py
visualizer="/Users/megamot/Programming/Python/Phase-problem-modeling/executable/visualizer.py"

# saving params
save_folder="/Users/megamot/Programming/Python/Phase-problem-modeling/data/executable_views"

mode="CA"

dpi=100
figsize_width=10.4
figsize_height=4.8
px_size=5e-6
cmap="jet"
show_plot=0
save_plot=1


if [ "$1" != "" ]
then
  echo "$1"
  $python $visualizer --file_path "$1" --save_folder $save_folder --mode $mode --dpi $dpi --figsize $figsize_width $figsize_height --px_size $px_size --cmap $cmap --show_plot $show_plot --save_plot $save_plot
fi

# pause