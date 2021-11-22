@echo off

rem path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"

rem path to executable python script
set script="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\slicer.py"

REM 'ARBITRARY', 'ENERGY_CENTER'
set mode="ARBITRARY"
set x=512
set y=256
set step=1

:loop
if "%~1" neq "" (
  %python% %script% --file_path "%~1" --mode %mode% --x %x% --y %y% --step %step%

  shift
  goto :loop
)

pause