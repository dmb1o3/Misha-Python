from __future__ import print_function

import numpy as np
import time
from rps import RPS
from matplotlib import pyplot as plt
from scipy.io import savemat

# Choose a method
METHOD = RPS.L2_SOLVER    # Least-squares
#METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
#METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
#METHOD = RPS.RPCA_SOLVER    # Robust PCA

# Choose a dataset
DATA_FOLDERNAME = './data/targetDownscaled/'    # Lambertian diffuse with cast shadow

LIGHT_FILENAME = '../../Downloads/2-10-25-RPSUpdate/RobustPhotometricStereo/data/targetDownscaled/lights_mm.txt'
MASK_FILENAME = '../../Downloads/2-10-25-RPSUpdate/RobustPhotometricStereo/data/targetDownscaled/mask/target_mask.tiff'


def generate_normal_map(directory):
    rps = RPS()
    rps.load_mask(filename=directory + "mask/target_mask.tiff")    # Load mask image
    rps.load_lighttxt(filename=directory + "calibrated_light.txt")    # Load light matrix
    rps.load_images(foldername=directory + "target/", ext='tiff')    # Load observations
    start = time.time()
    rps.solve(METHOD)    # Compute
    elapsed_time = time.time() - start
    print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
    rps.save_normalmap(filename=directory + "downscaledTargetNormals")    # Save the estimated normal map

    normals = np.load(directory + "/downscaledTargetNormals.npy")
    mat = {'Normals' : normals}
    savemat(directory + 'downscaledTargetNormals.mat', mat)
    plt.imshow(normals, cmap='gray')
    plt.show()

    return normals