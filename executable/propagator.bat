echo wave params
set wavelength=555e-9
set px_size=5e-6
set wave_path=%1

echo propagation params
set start=0e-3
set stop=100e-3
set step=10e-3
set method="angular_spectrum_band_limited"

echo saving params
set save_folder="%~dp1\%~n1 propagation"
set separate_save=1

echo path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"

echo path to propagator.py
set propagator="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\propagator.py"

echo running...
%python% %propagator% --wavelength %wavelength%  --px_size %px_size% --wave_path %wave_path% --start %start% --stop %stop% --step %step% --method %method% --save_folder %save_folder% --separate_save %separate_save%

pause
