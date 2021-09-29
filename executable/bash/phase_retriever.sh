#!/bin/bash

# wave params
dz=0
wavelength=555e-9
px_size=5e-6

# propagation params
solver="dct_1d"
bc="None"
threshold=0.135

# saving params
save_folder="/Users/megamot/Programming/Python/Phase-problem-modeling/data/executable_phases"

# path to python
python="/usr/local/bin/python3"
# path to phase_retriever.py
phase_retriever="/Users/megamot/Programming/Python/Phase-problem-modeling/executable/phase_retriever.py"

if [ "$1" != "" ]
then
  echo "$1"
  echo "$2"
  $python $phase_retriever --i1_path "$1" --i2_path "$2" --save_folder $save_folder --wavelength $wavelength  --px_size $px_size --dz $dz --solver $solver --bc $bc --threshold $threshold
fi

# pause
