@echo off

rem path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"

rem path to visualizer.py
set converter="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\convert2mat.py"


:loop
if "%~1" neq "" (
  echo %~1
  %python% %converter% --file_path "%~1"
  echo done

  shift
  goto :loop
)

pause
