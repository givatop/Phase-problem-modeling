@echo off

rem path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"

rem path to visualizer.py
set visualizer="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\visualizer.py"

rem path to true phase
set true_phase_file_path="C:\Users\IGritsenko\Desktop\phi z=0.000.npy"

set mode="ERROR"

set dpi=100
set figsize_width=12.4
set figsize_height=7.8
set px_size=5e-6
set cmap="jet"
set phase_ylabel="rad"
set show_plot=0
set save_plot=1

:loop
if "%~1" neq "" (
  echo %~1
  %python% %visualizer% --file_path "%~1" --true_phase_file_path %true_phase_file_path% --save_folder "%~dp1\" --mode %mode% --dpi %dpi% --figsize %figsize_width% %figsize_height% --px_size %px_size% --cmap %cmap% --show_plot %show_plot% --save_plot %save_plot% --phase_ylabel %phase_ylabel%
  echo done

  shift
  goto :loop
)

rem pause
