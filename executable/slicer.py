import os
import sys
import argparse
import numpy as np
from icecream import ic

sys.path.append(r'C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling')
from src.propagation.utils.math.general import row_slice, col_slice
from src.propagation.presenter.loader import load_file


# region Parser Arguments
ARBITRARY_MODE = 'ARBITRARY'
ENERGY_CENTER_MODE = 'ENERGY_CENTER'
NPY_EXTENSION = '.npy'

parser = argparse.ArgumentParser(description='Propagate initial wave on desired distances')

parser.add_argument(
    '--mode',
    type=str,
    choices=[ARBITRARY_MODE, ENERGY_CENTER_MODE],
    required=True,
    help='Режим'
)
parser.add_argument(
    '--file_path',
    type=str,
    required=True,
)
parser.add_argument(
    '--x',
    type=int,
    default=-1,
    help='-1 means no x-slice will be create'
)
parser.add_argument(
    '--y',
    type=int,
    default=-1,
    help='-1 means no y-slice will be create'
)
parser.add_argument(
    '--step',
    type=int,
    default=1,
)

args = parser.parse_args()

folder, filename = os.path.split(args.file_path)
filename = os.path.splitext(filename)[0]

# endregion

array = load_file(args.file_path)
height, width = array.shape

if args.mode == ARBITRARY_MODE:
    row, col = args.y, args.x
    rowslice = row_slice(array, row, args.step) if row != -1 else None
    colslice = col_slice(array, col, args.step) if col != -1 else None

elif args.mode == ENERGY_CENTER_MODE:
    ar_min, ar_max = np.min(array), np.max(array)

    if ar_max > abs(ar_min):
        argmax = np.argmax(array, axis=None)
    else:
        argmax = np.argmin(array, axis=None)

    row, col = np.unravel_index(argmax, array.shape)
    rowslice = row_slice(array, row, args.step)
    colslice = col_slice(array, col, args.step)

if rowslice is not None:
    save_filename = f'xslice y={row} {filename}{NPY_EXTENSION}'
    save_path = os.path.join(folder, save_filename)
    np.save(save_path, rowslice)

if colslice is not None:
    save_filename = f'yslice x={col} {filename}{NPY_EXTENSION}'
    save_path = os.path.join(folder, save_filename)
    np.save(save_path, colslice)

ic(args.file_path)
# for k, v in vars(args).items():
#     print(f'{k}: {v}')
