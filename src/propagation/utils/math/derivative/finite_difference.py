import math
import numpy as np


def central_finite_difference(planes: tuple, h: float = 1., deriv_order: int = 1) -> np.ndarray:
    """
    Central finite difference
    https://en.wikipedia.org/wiki/Finite_difference_coefficient#cite_note-fornberg-1
    https://github.com/maroba/findiff/tree/20194621fc2d54a10057cd6c3a9888eee67ab1f6
    """
    coefs = coefficients_grid(deriv_order, len(planes))

    # Удаление коэффициента, соответствующего нулевому члену
    coefs = np.delete(coefs, len(coefs)//2)

    result = list(plane * coefs[count] / h for count, plane in enumerate(planes))
    return sum(result)


def coefficients_grid(deriv_order: int, accuracy: int) -> np.ndarray:
    """
    Функция определения конечно-разностных коэффициентов
    для произвольного порядка производной и количества плоскостей расчёта
    """

    # Определение сетки для коэффициентов
    num_central = 2 * math.floor((deriv_order + 1) / 2) - 1 + accuracy
    num_side = num_central // 2
    offsets = list(range(-num_side, num_side + 1))

    # Определение коэффициентов
    center = _calc_coefs(deriv_order, offsets)

    return center


def _build_rhs(offsets: list, deriv: int) -> np.ndarray:
    """Построение правой матрицы для линейной системы уравнений"""
    b = [0 for _ in offsets]
    b[deriv] = math.factorial(deriv)

    return np.array(b, dtype='float')


def _build_matrix(offsets: list) -> np.ndarray:
    """Построение матрицы линейной системы уравнений для определения конечно-разностных коэффициентов"""
    a = [([1 for _ in offsets])]  # todo Tuple cast????
    for i in range(1, len(offsets)):
        a.append([j ** i for j in offsets])

    return np.array(a, dtype='float')


def _calc_coefs(deriv: int, offsets: list) -> np.ndarray:
    """Решение системы линейных уравнений для определения конечно-разностных коэффициентов"""

    # Определение матриц - системы линейных уравнений
    matrix = _build_matrix(offsets)
    rhs = _build_rhs(offsets, deriv)

    # Решение системы линейных уравнений
    coefs = np.linalg.solve(matrix, rhs)

    return coefs


def central_4point(p_minus2, p_minus1, p_plus1, p_plus2, h: float = 1.):
    """

    :param p_minus2:
    :param p_minus1:
    :param p_plus1:
    :param p_plus2:
    :param h: Шаг между СОСЕДНИМИ точками!
    :return:
    """
    return (p_minus2 - 8 * p_minus1 + 8 * p_plus1 - p_plus2) / (12 * h)


def central_2point(p_minus, p_plus, h):
    """

    :param p_minus:
    :param p_plus:
    :param h: 2*h = distance(p_plus - p_minus)
    :return:
    """
    if isinstance(p_minus, np.ndarray) and isinstance(p_plus, np.ndarray):
        if p_minus.shape != p_plus.shape:
            raise ValueError(f"Arrays shapes must be equal: {p_minus.shape} != {p_plus.shape}")
        return (p_plus - p_minus) / (2 * h)
    else:
        raise NotImplementedError("Implemented only for ndarrays")


def forward_2point(p, p_plus, h):
    if isinstance(p, np.ndarray) and isinstance(p_plus, np.ndarray):
        if p.shape != p_plus.shape:
            raise ValueError(f"Arrays shapes must be equal: {p.shape} != {p_plus.shape}")
        return (p_plus - p) / h
    else:
        raise NotImplementedError("Implemented only for ndarrays")


def backward_2point(p_minus, p, h):
    if isinstance(p, np.ndarray) and isinstance(p_minus, np.ndarray):
        if p.shape != p_minus.shape:
            raise ValueError(f"Arrays shapes must be equal: {p.shape} != {p_minus.shape}")
        return (p - p_minus) / h
    else:
        raise NotImplementedError("Implemented only for ndarrays")
