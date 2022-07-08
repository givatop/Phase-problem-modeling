import os

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from src.propagation.presenter.loader import load_files


folder = r'\\hololab.ru\store\Рабочие папки K-Team\Гриценко\1. Работа\1. Проекты\2021 РНФ TIE\1. Данные\2. Экспериментальные\7. M^2\шаг 500 мкм'
filename_1 = '1.5.tif'

for sigma in [25]:
    for z in np.arange(2., 10.4, 0.5):
        filename_2 = f'{z:.1f}.tif'
        filepath_1 = os.path.join(folder, filename_1)
        filepath_2 = os.path.join(folder, filename_2)
        base_filename_1 = os.path.splitext(os.path.basename(filename_1))[0]
        base_filename_2 = os.path.splitext(os.path.basename(filename_2))[0]

        i1, i2 = load_files([filepath_1, filepath_2])
        fig4 = plt.figure(dpi=100, figsize=[16, 9])
        ax41, ax42, ax43, ax44 = fig4.add_subplot(2, 2, 1), \
                                 fig4.add_subplot(2, 2, 3), \
                                 fig4.add_subplot(2, 2, 2), \
                                 fig4.add_subplot(2, 2, 4)

        fig5 = plt.figure(dpi=100, figsize=[16, 9])
        ax51, ax52, ax53, ax54 = fig5.add_subplot(2, 2, 1), \
                                 fig5.add_subplot(2, 2, 3), \
                                 fig5.add_subplot(2, 2, 2), \
                                 fig5.add_subplot(2, 2, 4)

        if i1.ndim == 1:

            # step = 1
            # i1 = i1[::step]
            # i2 = i2[::step]

            i1_max_index = np.argmax(i1)
            i2_max_index = np.argmax(i2)
            shift = i1_max_index - i2_max_index
            i2_shifted = np.roll(i2, shift)

            # ax41.plot(i1, label=f'{filename_1} {i1_max_index}')
            # ax41.plot(i2, label=f'{filename_2} {i2_max_index}')
            # ax42.plot(i2 - i1)
            #
            # ax43.plot(i1, '-+', label=f'{filename_1} {i1_max_index}')
            # ax43.plot(i2_shifted, '-.', label=f"{filename_2} {np.argmax(i2_shifted)} shifted")
            # ax44.plot(i2_shifted - i1)
            #
            # [ax.legend() for ax in [ax41, ax43]]
            # [ax.grid() for ax in [ax41, ax42, ax43, ax44]]

            # fp = os.path.join(folder, f'shifted {base_filename_2}.npy')
            # np.save(fp, i2_shifted)

            ic(i1_max_index)
            ic(i2_max_index)
            ic(shift)

        elif i1.ndim == 2:

            # i1_convolved = i1
            # i2_convolved = i2


            i1_convolved = gaussian_filter(i1, sigma=sigma)
            i2_convolved = gaussian_filter(i2, sigma=sigma)

            i1_ymax, i1_xmax = np.unravel_index(np.argmax(i1_convolved, axis=None), i1.shape)
            i2_ymax, i2_xmax = np.unravel_index(np.argmax(i2_convolved, axis=None), i2.shape)

            xshift = i1_xmax - i2_xmax
            yshift = i1_ymax - i2_ymax

            i2_shifted = np.roll(i2_convolved, (yshift, xshift), axis=(0, 1))

            ax41.plot(i1_convolved[::, i1_xmax], label=f'{filename_1} ymax={i1_ymax}')
            ax41.plot(i2_convolved[::, i2_xmax], label=f'{filename_2} ymax={i2_ymax}')
            ax42.plot((i2_convolved - i1_convolved)[::, i1_xmax])

            ax43.plot(i1_convolved[::, i1_xmax], '-.', label=f'{filename_1}')
            ax43.plot(i2_shifted[::, i1_xmax], '--', label=f"{filename_2} yshift={yshift}")
            ax44.plot((i2_shifted - i1_convolved)[::, i1_xmax])
            ax43.axvline(x=i1_ymax, color='green', linestyle='dotted', label='max')
            ax44.axvline(x=i1_ymax, color='green', linestyle='dotted')

            ax51.plot(i1_convolved[i1_ymax, ::], label=f'{filename_1} xmax={i1_xmax}')
            ax51.plot(i2_convolved[i2_ymax, ::], label=f'{filename_2} xmax={i2_xmax}')
            ax52.plot((i2_convolved - i1_convolved)[i1_ymax, ::])

            ax53.plot(i1_convolved[i1_ymax, ::], '-.', label=f'{filename_1}')
            ax53.plot(i2_shifted[i1_ymax, ::], '--', label=f"{filename_2} xshift={xshift}")
            ax54.plot((i2_shifted - i1_convolved)[i1_ymax, ::])
            ax53.axvline(x=i1_xmax, color='green', linestyle='dotted', label='max')
            ax54.axvline(x=i1_xmax, color='green', linestyle='dotted')

            [ax.legend() for ax in [ax41, ax43, ax51, ax53]]
            [ax.grid() for ax in [ax41, ax42, ax43, ax44,
                                  ax51, ax52, ax53, ax54]]

            [ax.set_ylim([-0.5, 0.5]) for ax in [ax42, ax44, ax52, ax54]]
            [ax.set_ylim([0., 1.]) for ax in [ax41, ax43, ax51, ax53]]

            ax41.set_title('Initial Intensities')
            ax43.set_title('Shifted Intensities')
            ax51.set_title('Initial Intensities')
            ax53.set_title('Shifted Intensities')
            fig4.suptitle('y-slice')
            fig5.suptitle('x-slice')

            fp = os.path.join(folder, f'shifted sigma={sigma} {base_filename_1}.npy')
            np.save(fp, i1_convolved)
            fp = os.path.join(folder, f'shifted sigma={sigma} {base_filename_2}.npy')
            np.save(fp, i2_shifted)

            # fp = os.path.join(folder, f'shifted xslice {base_filename_1}.npy')
            # np.save(fp, i1[i1_ymax, ::])
            #
            # fp = os.path.join(folder, f'shifted yslice {base_filename_1}.npy')
            # np.save(fp, i1[::, i1_xmax])
            #
            # fp = os.path.join(folder, f'shifted xslice {base_filename_2}.npy')
            # np.save(fp, i2_shifted[i1_ymax, ::])
            #
            # fp = os.path.join(folder, f'shifted yslice {base_filename_2}.npy')
            # np.save(fp, i2_shifted[::, i1_xmax])

            fig4.tight_layout()

            fp = os.path.join(folder, f'xshift sigma={sigma} {base_filename_1} {base_filename_2}.png')
            fig4.savefig(fp, bbox_inches='tight', pad_inches=0.1)
            fp = os.path.join(folder, f'yshift sigma={sigma} {base_filename_1} {base_filename_2}.png')
            fig5.savefig(fp, bbox_inches='tight', pad_inches=0.1)

            plt.close(fig4)
            plt.close(fig5)

            ic(filename_2)
            # plt.show()
