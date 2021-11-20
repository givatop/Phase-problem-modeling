import os
import sys
import argparse
from scipy.io import savemat

sys.path.append(r'C:\Users\IGritsenko\Documents\Python Scripts\TIE v2\Phase-problem-modeling')
sys.path.append(r'/Users/megamot/Programming/Python/Phase-problem-modeling')
from src.propagation.presenter.loader import load_file


MAT_EXTENSION = '.mat'


# CLI routine
parser = argparse.ArgumentParser(description='Retrieve Phase via Transport-of-Intensity Equation')
parser.add_argument(
    '--file_path',
    type=str,
    required=True,
)
args = parser.parse_args()

# Load & Convert
file = load_file(args.file_path)
mdict = {'TIE_phase': file}

# Saving
folder, filename = os.path.split(args.file_path)
filename = os.path.splitext(filename)[0]
save_path = os.path.join(folder, f'{filename}{MAT_EXTENSION}')
savemat(save_path, mdict)
