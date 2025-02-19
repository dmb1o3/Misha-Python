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
from scipy.io import savemat

# Choose a method
METHOD = RPS.L2_SOLVER    # Least-squares
#METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
#METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
#METHOD = RPS.RPCA_SOLVER    # Robust PCA

# Choose a dataset
DATA_FOLDERNAME = './data/targetDownscaled/'    # Lambertian diffuse with cast shadow

LIGHT_FILENAME = './data/targetDownscaled/lights_mm.txt'
MASK_FILENAME = './data/targetDownscaled/mask/target_mask.tiff'

\

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

normals = np.load("./downscaledTargetNormals.npy")
mat = {'Normals' : normals}
savemat('downscaledTargetNormals.mat', mat)
plt.imshow(normals, cmap='gray')
plt.show()