import os
import subprocess
import numpy as np


path2bat = r"C:\Users\IGritsenko\Documents\Python Scripts\TIE v2" \
           r"\Phase-problem-modeling\executable\phase_retriever.bat"

folder = r'\\hololab.ru\store\Рабочие папки K-Team\Сцепуро\М2\12.02'

z0 = 20

for half_dz in range(1, 12, 1):

    z_p = z0 + half_dz
    z_m = z0 - half_dz

    filename1 = f'{z_m:.1f}.tif'
    filename2 = f'{z_p:.1f}.tif'

    path1 = os.path.join(folder, filename1)
    path2 = os.path.join(folder, filename2)

    subprocess.run([path2bat, path1, path2])
