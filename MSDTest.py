from __future__ import print_function

import numpy as np
import time
from rps import RPS
import psutil
from height_map import (
    estimate_height_map,
)  # Local file 'height_map.py' in this repository.
from matplotlib import pyplot as plt
from skimage import io

# Choose a method
METHOD = RPS.L2_SOLVER    # Least-squares
#METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
#METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
#METHOD = RPS.RPCA_SOLVER    # Robust PCA

# Choose a dataset
DATA_FOLDERNAME = './data/targetDownscaled/'    # Lambertian diffuse with cast shadow

LIGHT_FILENAME = './data/targetDownscaled/lights.txt'
MASK_FILENAME = '../../Downloads/2-10-25-RPSUpdate/RobustPhotometricStereo/data/targetDownscaled/mask/target_mask.tiff'

"""
# Photometric Stereo
rps = RPS()
rps.load_mask(filename=MASK_FILENAME)    # Load mask image
rps.load_lighttxt(filename=LIGHT_FILENAME)    # Load light matrix
rps.load_images(foldername=DATA_FOLDERNAME, ext='tiff')    # Load observations
start = time.time()
rps.solve(METHOD)    # Compute
elapsed_time = time.time() - start
print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
rps.save_normalmap(filename="./downscaledTargetNormals")    # Save the estimated normal map
"""

def generate_depth_map(directory):
    NORMAL_MAP_A_IMAGE: np.ndarray = np.load( directory + "downscaledTargetNormals.npy")
    start = time.time()
    heights = estimate_height_map(NORMAL_MAP_A_IMAGE, raw_values=True)
    elapsed_time = time.time() - start

    print("Depth Map Calculations: elapsed_time:{0}".format(elapsed_time) + "[sec]")

    figure, axes = plt.subplots(1, 2, figsize=(7, 3))
    _ = axes[0].imshow(NORMAL_MAP_A_IMAGE)
    _ = axes[1].imshow(heights)

    x, y = np.meshgrid(range(heights.shape[1]), range(heights.shape[0]))
    _, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    _ = axes.scatter(x, y, heights, c=heights)

    plt.show()