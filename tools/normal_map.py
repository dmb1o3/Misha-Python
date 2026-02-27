from __future__ import print_function

import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from tools.rps import RPS


# Choose a configuration method
METHOD = RPS.L2_SOLVER    # Least-squares (default)
#METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
#METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
#METHOD = RPS.RPCA_SOLVER    # Robust PCA


def generate_normal_map(directory):
    """
    Given a directory containing a

    - Mask located in directory/mask/target_mask.tiff
    - Text file with light vectors in directory/calibrated_light.txt
    - Target images located in directory/target/*.tiff

    Will generate and save normal map at directory/downscaledTargetNormals.npy

    :param directory: Directory containing mask, light vectors and target images for normal map
    :return: None
    """
    rps = RPS() # Initialize RPS solver

    # Load inputs
    rps.load_mask(filename=f"{directory}mask/target_mask.tiff")  # Load mask image
    rps.load_lighttxt(filename=f"{directory}calibrated_light.txt")  # Load light vectors
    rps.load_images(foldername=f"{directory}target/", ext='tiff')  # Load target observations

    # Compute normals
    start_time = time.time()  # Start timer
    rps.solve(METHOD)
    elapsed_time = time.time() - start_time  # Stop timer and print out how long it took
    print(f"Photometric stereo completed in {elapsed_time:.4f} seconds")

    # Save raw normal map ([-1, 1] range)
    rps.save_normalmap(filename=f"{directory}normals/normal_map")

    # Prepare visualization (scale to [0, 255])
    normals = np.load(f"{directory}normals/normal_map.npy")  # Load normal map

    # Scale from [-1, 1] → [0, 255]
    normals = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)

    # Force Z (blue channel) to [128, 255] to avoid dark-looking normals
    normals[:, :, 2] = np.clip(normals[:, :, 2], 128, 255)  # Assuming your normal map has shape (height, width, 3)

    # Convert RGB → BGR for OpenCV
    cv2_normals = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)

    # Save and display result
    cv2.imwrite(directory + "normals/normal_map.png", cv2_normals)
    plt.imshow(normals)
    plt.show()

