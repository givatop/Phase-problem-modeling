@echo off

rem path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"

rem path to visualizer.py
set visualizer="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\visualizer.py"

set mode="ARRAY"


:loop
if "%~1" neq "" (
  echo %~1
  %python% %visualizer% --file_path "%~1" --save_folder "%~dp1\" --mode %mode%
  echo done

  shift
  goto :loop
)

pause
