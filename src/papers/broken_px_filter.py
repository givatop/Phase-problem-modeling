import os

import numpy as np
from PIL import Image
from src.propagation.presenter.loader import load_file
from src.propagation.utils.math.general import normalize

folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа' \
         r'\1. Проекты\2022 Vortex\2. Эксперимент\2022 март 28'

distances = np.arange(100, 101, 1)

for z in distances:

    # Загружаем из файла
    """
    z=0.{z:0>3.0f}
    z={z:.0f}.{"":0>3}
    """
    filename = f'z={z:.0f}.{"":0>3}.tif'  #
    path = os.path.join(folder, filename)
    array = load_file(path)

    print(filename)

    # Вырезаем нужную область
    array[496:536, 754:784] = array[495, 750]

    # Нормировка на 126 бит
    array = normalize(array, old_min=0, old_max=1.,
                      new_min=0, new_max=2 ** 16 - 1,
                      dtype=np.uint16)
    image = Image.fromarray(array, mode='I;16')

    # Сохранение в файл
    filename = f'f {filename}'
    path = os.path.join(folder, filename)
    image.save(path)
