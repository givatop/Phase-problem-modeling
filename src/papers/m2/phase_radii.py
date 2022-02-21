import os

import numpy as np
import matplotlib.pyplot as plt
from src.propagation.presenter.loader import load_file
import src.propagation.utils.math.units as units

plt.style.use('seaborn')

dzs = [60, 70, 80, 90, 100, 110, 120]

for dz in dzs:
    folder = rf'\\hololab.ru\store\Рабочие папки K-Team\Сцепуро\1. Работа\ФИАН\06. TIE\1. М2\5. Эксперименты\12.09\dz={dz}'

    DPI = 100
    FIGSIZE = [6.4, 4.8]
    WAVELENGTH = units.nm2m(515)
    WAVENUM = 2 * np.pi / WAVELENGTH
    PX_SIZE = units.um2m(5.04)
    PARABOLIC_DEGREE = 2

    xradii = []
    yradii = []
    zs = []

    i_max = 143
    z1s = np.arange(1, i_max, 1)
    z2s = np.arange(1 + dz, i_max, 1)

    print(f'z1 | z2 | z0 | R, mm, mm')

    with open(os.path.join(folder, f'phase radii dz={dz}mm.txt'), mode='wt') as txt:
        txt.write(f'parabola: a * x^2 + b * x + c\n\n')
        txt.write(f'{"z":>2} | {"Rx":^5} | {"Ry, mm":^7} | '
                  f'{"xa":^11} | {"xb":^9} | {"xc":^7} |    '
                  f'{"ya":^11} | {"yb":^9} | {"yc":^7} \n')

    for (z1, z2) in zip(z1s, z2s):

        z = (z1 + z2) // 2

        # Загрузка файлов сечений
        xfilename = f'xslice TIE sigma=0.8 f {z1:.1f} sigma=0.8 f {z2:.1f} dz={dz:.3f}mm.npy'
        yfilename = f'yslice TIE sigma=0.8 f {z1:.1f} sigma=0.8 f {z2:.1f} dz={dz:.3f}mm.npy'

        xfilepath = os.path.join(folder, xfilename)
        yfilepath = os.path.join(folder, yfilename)

        xbase_filename = os.path.splitext(os.path.basename(xfilename))[0]
        ybase_filename = os.path.splitext(os.path.basename(yfilename))[0]

        xslice = load_file(xfilepath)
        yslice = load_file(yfilepath)

        # Обрезаем всё лишнее
        xr, yr = 10, 10

        # xslice -= np.min(xslice)
        # yslice -= np.min(yslice)

        if np.max(xslice) > abs(np.min(xslice)):
            xargmax = np.argmax(xslice)
        else:
            xargmax = np.argmin(xslice)

        if np.max(yslice) > abs(np.min(yslice)):
            yargmax = np.argmax(yslice)
        else:
            yargmax = np.argmin(yslice)

        xslice = xslice[xargmax - xr:xargmax + xr]
        yslice = yslice[yargmax - xr:yargmax + yr]

        # Сетка
        height, width = yslice.shape[0], xslice.shape[0],
        x = np.arange(-width//2, width//2) * PX_SIZE
        y = np.arange(-height//2, height//2) * PX_SIZE

        # Fitting
        xa, xb, xc = np.polyfit(x, xslice, PARABOLIC_DEGREE)
        ya, yb, yc = np.polyfit(y, yslice, PARABOLIC_DEGREE)

        xfitted = np.poly1d([xa, xb, xc])(x)
        yfitted = np.poly1d([ya, yb, yc])(y)

        # Радиус кривизны ВФ
        xradius = WAVENUM / (2 * xa)
        yradius = WAVENUM / (2 * ya)

        # # Графики
        # fig1, ax1 = plt.subplots(nrows=1, ncols=1, dpi=DPI, figsize=FIGSIZE)
        # fig2, ax2 = plt.subplots(nrows=1, ncols=1, dpi=DPI, figsize=FIGSIZE)
        #
        # ax1.plot(xslice, linestyle='solid', label=f'Initial z={z} mm', color='orange')
        # ax2.plot(yslice, linestyle='solid', label=f'Initial z={z} mm', color='orange')
        #
        # ax1.plot(xfitted, linestyle='dashdot', label=f'Fitted R={units.m2mm(xradius):.2f} mm')
        # ax2.plot(yfitted, linestyle='dashdot', label=f'Fitted R={units.m2mm(yradius):.2f} mm')
        #
        # ax1.set_xlabel('x, mm')
        # ax2.set_xlabel('y, mm')
        #
        # [ax.set_ylabel('phase, rad') for ax in [ax1, ax2]]
        # [ax.legend() for ax in [ax1, ax2]]
        #
        # xpath = os.path.join(folder, f'{xbase_filename} R.png')
        # ypath = os.path.join(folder, f'{ybase_filename} R.png')
        #
        # fig1.savefig(xpath)
        # fig2.savefig(ypath)
        #
        # # plt.show()
        #
        # plt.close(fig1)
        # plt.close(fig2)

        with open(os.path.join(folder, f'phase radii dz={dz}mm.txt'), mode='at') as txt:
            z_str = f'{z:.3f}'.replace('.', ',')
            xradius_str = f'{units.m2mm(xradius):.3f}'.replace('.', ',')
            yradius_str = f'{units.m2mm(yradius):.3f}'.replace('.', ',')
            txt.write(f'{z}\t{xradius_str}\t{yradius_str}\t{xa:.5f}\t{xb:.5f}\t{xc:.5f}\t{ya:.5f}\t{yb:.5f}\t{yc:.5f}\n')

        # Сохранение данных в списки
        zs.append(z)
        xradii.append(units.m2mm(xradius))
        yradii.append(units.m2mm(yradius))

        print(f'{z1:>2} | {z2} | {z:>2} | {units.m2mm(xradius):.2f} | {units.m2mm(yradius):.2f}')

    # Итоговый график радиусов
    # fig, ax = plt.subplots(nrows=1, ncols=1, dpi=DPI, figsize=FIGSIZE)
    #
    # ax.plot(zs, xradii, label='x-radius')
    # ax.plot(zs, yradii, label='y-radius')
    #
    # ax.set_xlabel('z, mm')
    # ax.set_ylabel('radius, mm')
    # ax.set_title(f'dz = {dz} mm')
    # ax.legend()
    # ax.set_yscale('symlog')
    # # ax.set_xlim([-2, 50])
    # # ax.set_ylim([60, 130])
    #
    # path = os.path.join(folder, f'radii dz={dz} mm.png')
    # fig.savefig(path)
    # plt.close(fig)

    # Сохранение данных в файл
    path = os.path.join(folder, f'radii dz={dz} mm zs.npy')
    np.save(path, np.array(zs))

    path = os.path.join(folder, f'radii dz={dz} mm xradii.npy')
    np.save(path, np.array(xradii))

    path = os.path.join(folder, f'radii dz={dz} mm yradii.npy')
    np.save(path, np.array(yradii))

# plt.show()
