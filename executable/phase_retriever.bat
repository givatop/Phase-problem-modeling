echo wave params
set dz=10e-3
set wavelength=555e-9
set px_size=5e-6
set i1_path=%1
set i2_path=%2

echo propagation params
set solver="fft_1d"
set bc="None"
set threshold=0.1

echo saving params
set save_folder="%~dp1\%~n1 TIE"

echo path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"

echo path to phase_retriever.py
set phase_retriever="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\phase_retriever.py"

echo running...
%python% %phase_retriever% --wavelength %wavelength%  --px_size %px_size% --dz %dz% --i1_path %i1_path% --i2_path %i2_path% --solver %solver% --bc %bc% --threshold %threshold% --save_folder %save_folder%

pause
