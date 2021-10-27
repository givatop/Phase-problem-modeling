from math import pi


# В метры
def mm2m(mm):
    return mm * 1e-3


def um2m(um):
    return um * 1e-6


def nm2m(nm):
    return nm * 1e-9


def px2m(px, px_size_m=5.04e-6):
    return px * px_size_m


def rad2m(rad, wave_len_m):
    wave_num = 2 * pi / wave_len_m
    return rad / wave_num


def m2rad(m, wave_len_m):
    wave_num = 2 * pi / wave_len_m
    return m * wave_num


# В миллиметры
def m2mm(m):
    return m * 1e+3


def um2mm(um):
    return um * 1e-3


def nm2mm(nm):
    return nm * 1e-6


def px2mm(px, px_size_m=5.04e-6):
    return px * m2mm(px_size_m)


def rad2mm(rad, wave_len_m):
    wave_num = 2 * pi / m2mm(wave_len_m)
    return rad / wave_num


# В микрометры
def m2um(m):
    return m * 1e+6


def mm2um(mm):
    return mm * 1e+3


def nm2um(nm):
    return nm * 1e-3


# --> нм
def m2nm(m):
    return m * 1e+9


# --> %
def percent2decimal(percent):
    return percent / 100


# % -->
def decimal2percent(decimal):
    return decimal * 100


# В пиксели
def m2px(m, px_size_m=5.04e-6):
    return m / px_size_m


# Градусная мера
def degree2rad(degree):
    return degree * pi / 180


# Градусная мера
def rad2degree(rad):
    return rad * 180 / pi


def radians(value):
    return value


def degree(value):
    return value


def m(value):
    return value


def mm(value):
    return value


def um(value):
    return value


def nm(value):
    return value


def px(value):
    return value


if __name__ == '__main__':
    print(degree2rad(180))
    print(degree2rad(360))
    print(rad2degree(pi))
    print(rad2degree(4 * pi))
