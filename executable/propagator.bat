@echo off

rem wave params
set wavelength=555e-9
set px_size=5e-6
set wave_path=%1

rem propagation params
set start=-0.000004
set stop=0.000004
set step=1e-6
set method="angular_spectrum_band_limited"

rem saving params
set save_folder="%~dp1\%~n1 propagation"
set separate_save=1

rem path to python
set python="C:\Users\NStsepuro\PycharmProjects\TIE_NStsepuro\Phase-problem-modeling\Scripts\python.exe"

rem path to propagator.py
set propagator="C:\Users\NStsepuro\PycharmProjects\TIE_NStsepuro\Phase-problem-modeling\executable\propagator.py"

echo running...
%python% %propagator% --wavelength %wavelength%  --px_size %px_size% --wave_path %wave_path% --start %start% --stop %stop% --step %step% --method %method% --save_folder %save_folder% --separate_save %separate_save%

rem pause
