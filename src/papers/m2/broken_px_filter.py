import os

import numpy as np
from PIL import Image
from src.propagation.presenter.loader import load_file
from src.propagation.utils.math.general import normalize

folder = r'\\hololab.ru\store\Рабочие папки K-Team\Сцепуро\М2\12.09'

dzs = np.arange(0, 144)

for dz in dzs:

    print(f'dz: {dz:.3f} mm')

    # Загружаем из файла
    filename = f'{dz:0>3.0f}.tif'
    path = os.path.join(folder, filename)
    array = load_file(path)

    # Вырезаем нужную область
    array[490:540, 740:800] = array[490, 750]

    # Нормировка на 126 бит
    array = normalize(array, old_min=0, old_max=1.,
                      new_min=0, new_max=2 ** 16 - 1,
                      dtype=np.uint16)
    image = Image.fromarray(array, mode='I;16')

    # Сохранение в файл
    filename = f'f {dz:.1f}.tif'
    path = os.path.join(folder, filename)
    image.save(path)
