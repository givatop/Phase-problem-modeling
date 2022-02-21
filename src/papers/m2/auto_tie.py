import os
import subprocess
import numpy as np


path2bat = r"C:\Users\IGritsenko\Documents\Python Scripts\TIE v2" \
           r"\Phase-problem-modeling\executable\phase_retriever.bat"

folder = r'\\hololab.ru\store\Рабочие папки K-Team\Сцепуро\1. Работа\ФИАН\06. TIE\1. М2\5. Эксперименты\12.09'

dz = 10
i_max = 143
z1s = np.arange(1, i_max, 1)
z2s = np.arange(1 + dz, i_max, 1)

sigma = 0.8

for (z1, z2) in zip(z1s, z2s):

    filename1 = f'sigma={sigma:.1f} f {z1:.1f}.npy'
    filename2 = f'sigma={sigma:.1f} f {z2:.1f}.npy'

    path1 = os.path.join(folder, filename1)
    path2 = os.path.join(folder, filename2)

    subprocess.run([path2bat, path1, path2])
