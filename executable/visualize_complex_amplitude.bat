echo path to python
set python="C:\Users\IGritsenko\.virtualenvs\Phase-problem-modeling-VD4lAbA6\Scripts\python.exe"

echo path to visualizer.py
set visualizer="C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling\executable\visualizer.py"

rem choices=['complex_amplitude', 'slice', 'error', 'tie_phase'],
set mode="complex_amplitude"

set dpi=100
set figsize_width=9.4
set figsize_height=4.8
set px_size=5e-6
set cmap="jet"
set show_plot=0
set save_plot=1


rem обход по всем файлам
for	%%f in (%*) do call :loop %%f


:loop
rem путь к файлу
set file_path=%1
rem папка куда сохраняем
set save_folder="%~dp1\"


echo running...
%python% %visualizer% --file_path %file_path% --save_folder %save_folder% --mode %mode% --dpi %dpi% --figsize %figsize_width% %figsize_height% --px_size %px_size% --cmap %cmap% --show_plot %show_plot% --save_plot %save_plot%

rem pause

exit /b
