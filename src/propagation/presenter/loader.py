import os
import numpy as np

from typing import List
from PIL import Image
from src.propagation.utils.math.general import normalize


NDARRAY_EXTENSION = '.npy'
TXT_EXTENSION = '.txt'


def load_files(paths: List[str]) -> List[np.ndarray]:
    return [load_file(path) for path in paths]


def load_file(path: str) -> np.ndarray:
    _, extension = os.path.splitext(path)
    if extension == NDARRAY_EXTENSION:
        array = np.load(path)
    elif extension == TXT_EXTENSION:
        array = load_txt(path)
    else:
        array = load_image(path)
    return array


def load_txt(path: str) -> np.ndarray:
    """
    Загрузка массива данных с помощью numpy.loadtxt.
    Поддержка текстовых файлов из ПО Fresnel.
    :param path: путь к файлу
    :return:
    """
    try:
        array = np.loadtxt(path)
    except ValueError:
        with open(path) as f:
            lines = f.readlines()

        lines.pop(0)  # cut first row without data
        lines = [line.strip().split('\t')[1:] for line in lines]  # save Y-values col without X-values col
        [line.append('0.0') for line in lines if len(line) == 0]  # Fresnel write nothing instead of zero in the phase txt file
        array = np.asarray(lines, dtype=float)
        if array.shape[1] == 1:
            array = array.reshape([array.shape[0]])  # "transpose" col-vector to row-vector

    return array


def load_image(path: str) -> np.ndarray:
    """
    Загружает изображение, конвертирует его в numpy.ndarray (dtype=np.float64), приводит к динамическому диапазону
    [0.0 ... 1.0].
    Цветные изображения конвертируются в полутоновые.
    :param path: путь к файлу
    :return матрица
    """
    gray_8bit = 'L'
    gray_16bit = 'I;16'

    img = Image.open(path)
    gray_mode = img.mode

    if gray_mode == gray_8bit:
        old_max = 2 ** 8 - 1  # 255

    elif gray_mode == gray_16bit:
        old_max = 2 ** 16 - 1  # 65 535

    else:  # color-image
        img = img.convert(gray_8bit)
        old_max = 2 ** 8 - 1

    return normalize(np.asarray(img, np.float64), old_min=0, old_max=old_max)


if __name__ == '__main__':
    path = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\11. GAUSS\Fresnel\Новая папка\phi z=0.000.txt'
    f = load_file(path)
    print(abs(f.min()) + abs(f.max()))
