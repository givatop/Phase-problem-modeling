echo path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"

echo path to visualizer.py
set visualizer="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\visualizer.py"

rem choices=['complex_amplitude', 'slice', 'error', 'tie_phase'],
set mode="tie_phase"

set dpi=100
set figsize_width=8.4
set figsize_height=4.8
set px_size=5e-6
set cmap="jet"


rem обход по всем файлам
for	%%f in (%*) do call :loop %%f


:loop
rem путь к файлу
set file_path=%1
rem папка куда сохраняем
set save_folder="%~dp1\%~n1 images"


echo running...
%python% %visualizer% --file_path %file_path% --save_folder %save_folder% --mode %mode% --dpi %dpi% --figsize %figsize_width% %figsize_height% --px_size %px_size% --cmap %cmap%

pause

exit /b
