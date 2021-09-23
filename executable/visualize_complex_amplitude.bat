@echo off
rem path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"

rem path to visualizer.py
set visualizer="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\visualizer.py"

set mode="CA"

set dpi=100
set figsize_width=10.4
set figsize_height=4.8
set px_size=5e-6
set cmap="jet"
set show_plot=0
set save_plot=1


:loop
if "%~1" neq "" (
  echo %~1
  %python% %visualizer% --file_path "%~1" --save_folder "%~dp1\" --mode %mode% --dpi %dpi% --figsize %figsize_width% %figsize_height% --px_size %px_size% --cmap %cmap% --show_plot %show_plot% --save_plot %save_plot%

  shift
  goto :loop
)

rem pause