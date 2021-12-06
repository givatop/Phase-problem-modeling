@echo off
rem wave params
set dz=0
set wavelength=555e-9
set px_size=5e-6

rem 'fft_1d', 'fft_2d', 'dct_2d', 'dct_1d'
set solver="dct_1d"

rem # 'PBC', 'NBC', 'DBC', 'None'
set bc="None"
set threshold=0.1

rem path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"
rem path to phase_retriever.py
set phase_retriever="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\phase_retriever.py"

rem calculate radius of curvature
set radius=1

:loop
if "%~1" neq "" (
  rem echo %~1
  rem echo %~2
  %python% %phase_retriever% --i1_path "%~1" --i2_path "%~2" --save_folder "%~dp1\" --wavelength %wavelength%  --px_size %px_size% --dz %dz% --solver %solver% --bc %bc% --threshold %threshold% --radius %radius%

  shift
  shift
  goto :loop
)


rem pause
