from __future__ import print_function

import cv2
from matplotlib import pyplot as plt
from rps import RPS
import time
import numpy as np


# Choose a method
METHOD = RPS.L2_SOLVER    # Least-squares
#METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
#METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
#METHOD = RPS.RPCA_SOLVER    # Robust PCA


def generate_normal_map(directory):
    """
    Given a directory containing a

    - Mask located in directory/mask/target_mask.tiff
    - A text file with light vectors in directory/calibrated_light.txt
    - Target images located in directory/target/

    Will generate and save normal map at directory/downscaledTargetNormals.npy

    :param directory: Directory containing mask, light vectors and target images for normal map
    :return: Does not return anything
    """
    rps = RPS()
    # Load mask image
    rps.load_mask(filename=directory + "mask/target_mask.tiff")
    # Load light vectors
    rps.load_lighttxt(filename=directory + "calibrated_light.txt")
    # Load target observations
    rps.load_images(foldername=directory + "target/", ext='tiff')
    # Start timer
    start = time.time()
    # Compute normal map
    rps.solve(METHOD)
    # Stop timer and print out how long it took
    elapsed_time = time.time() - start
    print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
    # Save the estimated normal map
    rps.save_normalmap(filename=directory + "normals/normal_map")
    # Load normal map
    normals = np.load(directory + "normals/normal_map.npy")
    # Scale normal map from [-1, 1] to [0, 1] and then display it. Just so it displays properly
    #normals = (normals + 1.0) / 2.0
    # Scale normal map from [-1, 1] to [0, 255] and then display it. Just so it displays properly
    normals = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    cv2.imwrite(directory + "normals/normal_map.png", normals)
    plt.imshow(normals, cmap='gray')
    plt.show()

