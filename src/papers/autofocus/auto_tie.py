import os
import subprocess
import numpy as np
import src.propagation.utils.math.units as units


path2bat = r"C:\Users\IGritsenko\Documents\Python Scripts\TIE v2" \
           r"\Phase-problem-modeling\executable\phase_retriever.bat"

folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа' \
         r'\1. Проекты\2021 РНФ TIE\2. Теория\Г2.1 автофокус\2. Автофокусировка' \
         r'\z=0.000 propagation'

distances = units.um2mm(np.arange(40, 1001, 40))

for dz in distances:
    z1 = -dz / 2
    z2 = dz / 2

    filename1 = f'i z={z1:.3f}.npy'
    filename2 = f'i z={z2:.3f}.npy'

    path1 = os.path.join(folder, filename1)
    path2 = os.path.join(folder, filename2)

    subprocess.run([path2bat, path1, path2])
