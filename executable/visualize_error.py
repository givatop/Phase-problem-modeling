import os
import numpy as np
import matplotlib.pyplot as plt

from src.propagation.utils.math.units import m2mm, mm2m, px2mm


path = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\1. Тестовые\1. FFT 1D\i=rect0.5 phi=sphere propagation\z=0.000 propagation'
fn1 = 'phi z=0.000.npy'
tie_fn = 'TIE i z=0.500 i z=-0.500 dz=-1.000mm.npy'

dz = 1000e-6
px_size = 5e-6
wavelength = 555e-9
threshold = 0.1

phase1 = np.load(os.path.join(path, fn1))
phase_tie = np.load(os.path.join(path, tie_fn))

fig = plt.figure()
ax = fig.gca()

width = phase1.shape[0]
x = px2mm(np.arange(-width//2, width//2), px_size)
start, stop = width//2 - width//20, width//2 + width//20
ax.plot(x[start:stop], phase1[start:stop], '*', label='z = 0')
ax.plot(x[start:stop], (phase_tie + (phase1.max() - phase_tie.max()))[start:stop], label=f'TIE dz={m2mm(dz):.3f} mm')

ax.grid()
ax.legend()
ax.set_xlabel('x, mm')
ax.set_ylabel('phase, rad')
# ax.set_ylim([365, 370])

filename = f'{tie_fn[:-4]} ERROR.png'
filepath = os.path.join(path, filename)
fig.savefig(filepath)

# plt.show()
# np.save(os.path.join(path, f'TIE phase.npy'), phase)
