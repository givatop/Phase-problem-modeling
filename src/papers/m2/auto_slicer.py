import os
import subprocess
import numpy as np


path2bat = r"C:\Users\IGritsenko\Documents\Python Scripts\TIE v2" \
           r"\Phase-problem-modeling\executable\slicer.bat"

dzs = [10, 15, 20, 40, 50, 60, 70, 80, 90, 100, 110, 120]

for dz in dzs:
    folder = rf'\\hololab.ru\store\Рабочие папки K-Team\Сцепуро\1. Работа\ФИАН\06. TIE\1. М2\5. Эксперименты\12.09\dz={dz:.0f}'

    i_max = 143
    z1s = np.arange(1, i_max, 1)
    z2s = np.arange(1 + dz, i_max, 1)

    sigma = 0.8

    for (z1, z2) in zip(z1s, z2s):

        filename = f'TIE sigma={sigma:.1f} f {z1:.1f} sigma={sigma:.1f} f {z2:.1f} dz={dz:.3f}mm.npy'

        path = os.path.join(folder, filename)

        subprocess.run([path2bat, path])
