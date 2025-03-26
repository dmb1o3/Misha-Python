from __future__ import print_function
from rps import RPS
from calibrate_light import calibrate_light_test

from MyCalcs import (
    runTest
)
import argparse
import numpy as np
import time
import psutil
from height_map import (
    estimate_height_map,
)  # Local file 'height_map.py' in this repository.
from matplotlib import pyplot as plt
from skimage import io, transform
import cv2 as cv
from PIL import Image
from scipy.sparse import lil_matrix
from scipy.ndimage import convolve

# Choose a method
METHOD = RPS.L2_SOLVER    # Least-squares
#METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
#METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
#METHOD = RPS.RPCA_SOLVER    # Robust PCA

# Choose a dataset
DATA_FOLDERNAME = './data/target/'    # Lambertian diffuse with cast shadow
MASK_FOLDERNAME = './data/mask/'
LIGHT_FOLDERNAME = './data/light/'
RESULTS_FOLDERNAME = './data/results/'
IMAGES_NAME = 'innard_'
NORMAL_FILENAME = './data/results/monkey_normal.png' #'./data/results/innard_normal.jpg' # './data/normal_run/normal_mapping_a.png' target_normal.png
EXPORT_FILENAME = './data/results/export_normal.png'
EXPORT_FILENAME2 = './data/results/export_normal2.png'
EXPORT_FILENAME3 = './data/results/export_normal3.png'
EXPORT_FILENAME4 = './data/results/export_normal4.png'

HEIGHT_FILENAME = './data/results/export_normal4.png'

scale = 1.0 # Makes image smaller to improve performance for testing, should be 1 for actual runs

showImages = True
renderModel = True


def runPStereo(rps):
    # Photometric Stereo
    #rps = RPS()
    rps.load_mask(MASK_FOLDERNAME + IMAGES_NAME + 'mask.tiff')    # Load mask image
    rps.load_lighttxt(LIGHT_FOLDERNAME + IMAGES_NAME + 'light.txt')    # Load light matrix
    rps.load_images(foldername=DATA_FOLDERNAME, ext='tiff')    # Load observations
    start = time.time()
    rps.solve(METHOD)    # Compute
    elapsed_time = time.time() - start
    print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
    rps.save_normalmap(filename="./downscaledTargetNormals")    # Save the estimated normal map
    NORMAL_MAP_A_IMAGE: np.ndarray = np.load("./downscaledTargetNormals.npy")
    #NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE + 1) / 2
    #cv.imshow("hi", NORMAL_MAP_A_IMAGE)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE-np.min(NORMAL_MAP_A_IMAGE))/(np.max(NORMAL_MAP_A_IMAGE)-np.min(NORMAL_MAP_A_IMAGE))
    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE* 255).astype(np.uint8)

    #cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal_red.png', NORMAL_MAP_A_IMAGE[:, :, 0])
    #v.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal_green.png', NORMAL_MAP_A_IMAGE[:, :, 1])
    #cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal_blue.png', NORMAL_MAP_A_IMAGE[:, :, 2])

    #print(NORMAL_MAP_A_IMAGE.shape)
    #print(NORMAL_MAP_A_IMAGE[0][0])

    #cv.imshow("Name", NORMAL_MAP_A_IMAGE)

    #cv.waitKey(0)
    
    #cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal.png', NORMAL_MAP_A_IMAGE)

    image_bgr = cv.cvtColor(NORMAL_MAP_A_IMAGE, cv.COLOR_RGB2BGR)

    if showImages:
        cv.imshow("Name", image_bgr)
        cv.waitKey(0)

    cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal.png', image_bgr)
    #rps.save_normalmap_as_png(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal.png')
    
    
# For seeing the 3d model of a height map
def runHeightGen(rps):
    NORMAL_MAP_A_IMAGE: np.ndarray = np.load("./downscaledTargetNormals.npy")
    start = time.time()

    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE-np.min(NORMAL_MAP_A_IMAGE))/(np.max(NORMAL_MAP_A_IMAGE)-np.min(NORMAL_MAP_A_IMAGE))
    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE* 255).astype(np.uint8)
    image_bgr = cv.cvtColor(NORMAL_MAP_A_IMAGE, cv.COLOR_RGB2BGR)
    #image_bgr = cv.cvtColor(image_bgr, cv.COLOR_RGB2BGR)


    heights = estimate_height_map(image_bgr, raw_values=True)
    elapsed_time = time.time() - start

    #NORMAL_MAP_A_IMAGE = heights
   # NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE-np.min(NORMAL_MAP_A_IMAGE))/(np.max(NORMAL_MAP_A_IMAGE)-np.min(NORMAL_MAP_A_IMAGE))
    #NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE* 255).astype(np.uint8)
    #image_bgr = cv.cvtColor(NORMAL_MAP_A_IMAGE, cv.COLOR_RGB2BGR)
    #cv.imwrite('./data/results/' + 'something_' + 'height2.png', image_bgr)

    print("Depth Map Calculations: elapsed_time:{0}".format(elapsed_time) + "[sec]")

    figure, axes = plt.subplots(1, 2, figsize=(7, 3))
    _ = axes[0].imshow(cv.cvtColor(image_bgr, cv.COLOR_RGB2BGR))
    _ = axes[1].imshow(heights)

    #image_bgr = cv.cvtColor(NORMAL_MAP_A_IMAGE, cv.COLOR_RGB2BGR)

    cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal2.png', image_bgr)
    cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'heights.png', heights)

    if renderModel: # this causes massive lag
        x, y = np.meshgrid(range(heights.shape[1]), range(heights.shape[0]))
        _, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        _ = axes.scatter(x, y, heights, c=heights)

        plt.show()
    else:
        #cv.waitKey(0)
        plt.show()
        plt.waitforbuttonpress



def runHeightModel():
    heights: np.ndarray = cv.imread(HEIGHT_FILENAME)

    
    x, y = np.meshgrid(range(heights.shape[1]), range(heights.shape[0]))
    _, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    _ = axes.scatter(x, y, heights[:, :, 0], c=heights[:, :, 0])

   # cv.imwrite(EXPORT_FILENAME, (heights *255).astype(np.uint8))

    plt.show()


def runHeightGen2():

    NORMAL_MAP_A_IMAGE: np.ndarray = Image.open('./data/normal_run/normal_mapping_a.png')

    #NORMAL_MAP_A_IMAGE: np.ndarray = np.load("./downscaledTargetNormals.npy")
    start = time.time()

    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE-np.min(NORMAL_MAP_A_IMAGE))/(np.max(NORMAL_MAP_A_IMAGE)-np.min(NORMAL_MAP_A_IMAGE))
    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE* 255).astype(np.uint8)
    image_bgr = cv.cvtColor(NORMAL_MAP_A_IMAGE, cv.COLOR_RGB2BGR)
    #image_bgr = cv.cvtColor(image_bgr, cv.COLOR_RGB2BGR)


    heights = estimate_height_map(image_bgr, raw_values=True)
    elapsed_time = time.time() - start

    #NORMAL_MAP_A_IMAGE = heights
   # NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE-np.min(NORMAL_MAP_A_IMAGE))/(np.max(NORMAL_MAP_A_IMAGE)-np.min(NORMAL_MAP_A_IMAGE))
    #NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE* 255).astype(np.uint8)
    #image_bgr = cv.cvtColor(NORMAL_MAP_A_IMAGE, cv.COLOR_RGB2BGR)
    #cv.imwrite('./data/results/' + 'something_' + 'height2.png', image_bgr)

    print("Depth Map Calculations: elapsed_time:{0}".format(elapsed_time) + "[sec]")

    figure, axes = plt.subplots(1, 2, figsize=(7, 3))
    _ = axes[0].imshow(cv.cvtColor(image_bgr, cv.COLOR_RGB2BGR))
    _ = axes[1].imshow(heights)

    #image_bgr = cv.cvtColor(NORMAL_MAP_A_IMAGE, cv.COLOR_RGB2BGR)

    cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal2.png', image_bgr)
    cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'heights.png', heights)

    if renderModel: # this causes massive lag
        x, y = np.meshgrid(range(heights.shape[1]), range(heights.shape[0]))
        _, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        _ = axes.scatter(x, y, heights, c=heights)

        plt.show()
    else:
        #cv.waitKey(0)
        plt.show()
        plt.waitforbuttonpress


#def saveNormals():

def compute_height_map3():
    NORMAL_MAP_A_IMAGE: np.ndarray = np.load("./downscaledTargetNormals.npy")

    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE-np.min(NORMAL_MAP_A_IMAGE))/(np.max(NORMAL_MAP_A_IMAGE)-np.min(NORMAL_MAP_A_IMAGE))
    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE* 255).astype(np.uint8)
    image_bgr = cv.cvtColor(NORMAL_MAP_A_IMAGE, cv.COLOR_RGB2BGR)
    # Compute the gradients in x and y directions

    normal_map = image_bgr

    gradient_x = np.gradient(normal_map[:, :, 0], axis=0)
    gradient_y = np.gradient(normal_map[:, :, 1], axis=1)

    #print(normal_map.shape)
    #print(len(gradient_x))
    #print(len(gradient_y))
    #print(gradient_y[0][0])
    # Integrate the gradients to obtain height values
    height_map = np.zeros_like(normal_map[:, :, 0])
    for y in range(normal_map.shape[0]):
        for x in range(normal_map.shape[1]):
            #pass
            #print(np.sqrt(1 - gradient_x[y][x]**2 - gradient_y[y][x]**2))
            height_map[y][x] = np.sqrt(1 - gradient_x[y][x]**2 - gradient_y[y][x]**2)
    
    cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'heights.png', height_map)
    #return height_map

    figure, axes = plt.subplots(1, 2, figsize=(7, 3))
    _ = axes[0].imshow(normal_map)
    _ = axes[1].imshow(height_map)

    if renderModel: # this causes massive lag
        x, y = np.meshgrid(range(height_map.shape[1]), range(height_map.shape[0]))
        _, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        _ = axes.scatter(x, y, height_map, c=height_map)

        plt.show()


def compute_normal_map(rps):
    rps.load_mask(MASK_FOLDERNAME + IMAGES_NAME + 'mask.tiff')    # Load mask image
    rps.load_lighttxt(LIGHT_FOLDERNAME + IMAGES_NAME + 'light.txt')    # Load light matrix
    rps.load_images(foldername=DATA_FOLDERNAME, ext='tiff')    # Load observations
    start = time.time()
    rps.solve(METHOD)    # Compute
    elapsed_time = time.time() - start
    print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")

    rps.save_normalmap(filename="./downscaledTargetNormals")    # Save the estimated normal map
    map: np.ndarray = np.load("./downscaledTargetNormals.npy")
    if map.shape[2] == 3:
        new_column = np.full((map.shape[0], map.shape[1], 1), 1, dtype=np.uint8)
        map = np.concatenate((map, new_column), axis=2)

    map = (map-np.min(map))/(np.max(map)-np.min(map))
    map = (map* 255).astype(np.uint8)

    io.imsave(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal.png', map)


    if showImages:
        cv.imshow("Name", cv.cvtColor(map, cv.COLOR_RGB2BGR))
        cv.waitKey(0)



def compute_height_map():
    NORMAL_MAP_A_IMAGE_FIRST: np.ndarray = io.imread(NORMAL_FILENAME)

    #print(NORMAL_MAP_A_IMAGE_FIRST.shape)
    #print(NORMAL_MAP_A_IMAGE_FIRST[0, 0])

    #NORMAL_MAP_A_IMAGE = transform.rescale(NORMAL_MAP_A_IMAGE, 0.5, anti_aliasing=True)
    #NORMAL_MAP_A_IMAGE = NORMAL_MAP_A_IMAGE.resize(int(NORMAL_MAP_A_IMAGE))
    NORMAL_MAP_A_IMAGE = transform.resize(NORMAL_MAP_A_IMAGE_FIRST, (int(NORMAL_MAP_A_IMAGE_FIRST.shape[0] * scale), int(NORMAL_MAP_A_IMAGE_FIRST.shape[1] * scale)),
                                 anti_aliasing=True)
    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE * 255).astype(int)

    map = NORMAL_MAP_A_IMAGE
    #cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal_red.png', map[:, :, 0])
    #cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal_green.png', map[:, :, 1])
    #cv.imwrite(RESULTS_FOLDERNAME + IMAGES_NAME + 'normal_blue.png', map[:, :, 2])

    #print(NORMAL_MAP_A_IMAGE.shape)
    #print(NORMAL_MAP_A_IMAGE[0, 0])

    heights = estimate_height_map(NORMAL_MAP_A_IMAGE, raw_values=True)
    val =  np.max(heights) - np.min(heights)
    heights = (heights-np.min(heights))/(np.max(heights)-np.min(heights))
    heights = (heights * val)

    figure, axes = plt.subplots(1, 2, figsize=(7, 3))
    _ = axes[0].imshow(NORMAL_MAP_A_IMAGE)
    _ = axes[1].imshow(heights)

    x, y = np.meshgrid(range(heights.shape[1]), range(heights.shape[0]))
    _, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    _ = axes.scatter(x, y, heights, c=heights)

    plt.show()

def compute_height_map2():
    NORMAL_MAP_A_IMAGE_FIRST: np.ndarray = io.imread(NORMAL_FILENAME)
    NORMAL_MAP_A_IMAGE = transform.resize(NORMAL_MAP_A_IMAGE_FIRST, (int(NORMAL_MAP_A_IMAGE_FIRST.shape[0] * scale), int(NORMAL_MAP_A_IMAGE_FIRST.shape[1] * scale)),
                                 anti_aliasing=True)
    NORMAL_MAP_A_IMAGE = (NORMAL_MAP_A_IMAGE * 255).astype(int)
    map = NORMAL_MAP_A_IMAGE

    maskImage = io.imread(MASK_FOLDERNAME + IMAGES_NAME + 'mask.png')
    #print(maskImage.shape)
    maskImage = transform.resize(maskImage, (int(maskImage.shape[0] * scale), int(maskImage.shape[1] * scale)),
                                 anti_aliasing=True)
    maskImage = (maskImage * 255).astype(int)
    maskImage = cv.convertScaleAbs(maskImage)


    print(map.shape)
    print(maskImage.shape)
    maskImage = cv.cvtColor(maskImage, cv.COLOR_RGB2GRAY)

    heights = normal_to_height_kernel(NORMAL_MAP_A_IMAGE)
    val =  np.max(heights) - np.min(heights)
    heights = (heights-np.min(heights))/(np.max(heights)-np.min(heights))
    heights = (heights * val)

    #figure, axes = plt.subplots(1, 2, figsize=(7, 3))
    #_ = axes[0].imshow(NORMAL_MAP_A_IMAGE)
    #_ = axes[1].imshow(heights)

    #x, y = np.meshgrid(range(heights.shape[1]), range(heights.shape[0]))
    #_, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    #_ = axes.scatter(x, y, heights, c=heights)

    cv.imwrite(EXPORT_FILENAME, (heights *255).astype(np.uint8))

    #plt.show()


def normal_to_height_kernel(normal_map):
    # Design kernels to approximate partial derivatives in X, Y, and Z directions
    kernel_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    kernel_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
    kernel_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

    # Perform convolutions to extract height information in X, Y, and Z directions
    height_x = convolve(normal_map[:, :, 0], kernel_x)
    height_y = convolve(normal_map[:, :, 1], kernel_y)
    height_z = convolve(normal_map[:, :, 2], kernel_z)

    convx = (height_x-np.min(height_x))/(np.max(height_x)-np.min(height_x))
    convy = (height_y-np.min(height_y))/(np.max(height_y)-np.min(height_y))
    convz = (height_z-np.min(height_z))/(np.max(height_z)-np.min(height_z))

    cv.imwrite(EXPORT_FILENAME2, (convx *255).astype(np.uint8))
    cv.imwrite(EXPORT_FILENAME3, (convy *255).astype(np.uint8))
    cv.imwrite(EXPORT_FILENAME4, (convz *255).astype(np.uint8))


    # Combine the results to get the height map
    height_map = (height_x + height_z + height_z)/3

    return height_map


def normal_to_height_integration(normal_map):
    height_map = np.zeros(normal_map.shape[:2])

    for i in range(normal_map.shape[0]):
        for j in range(normal_map.shape[1]):
            # Integrate the normal map to get the height map
            height_map[i, j] = np.sum(normal_map[:i+1, :j+1, 2])

    cv.imwrite(EXPORT_FILENAME, (height_map *255).astype(np.uint8))
    return height_map

def compute_my_heights():
    normal_map: np.ndarray = io.imread(NORMAL_FILENAME)
    #print(normal_map.shape)
    #print(normal_map[0])
    normal_map = transform.resize(normal_map, (int(normal_map.shape[0] * scale), int(normal_map.shape[1] * scale)), anti_aliasing=True)
    normal_map = (normal_map * 255).astype(int)
    height_map = np.ones((normal_map.shape[0], normal_map.shape[1])) * 127
    #print(normal_map[0])
    for row in range(normal_map.shape[0]):  # Iterate over rows
        row_val = 0
        for col in range(normal_map.shape[1]):
            row_val += normal_map[row][col][0]/2.0 - 64
            height_map[row][col] += row_val
        #print(map[row])
        #print(height_map[row])
    print(height_map[0])
    height_map = (height_map-np.min(height_map))/(np.max(height_map)-np.min(height_map))
    print(height_map[20])
    cv.imwrite(EXPORT_FILENAME, (height_map *255).astype(np.uint8))

    height_map2 = np.ones((normal_map.shape[0], normal_map.shape[1])) * 127
    for col in range(normal_map.shape[1]):  # Iterate over rows
        col_val = 0
        for row in range(normal_map.shape[0]):
            col_val += normal_map[row][col][1]/2.0 - 64
            height_map2[row][col] += col_val
    height_map2 = (height_map2-np.min(height_map2))/(np.max(height_map2)-np.min(height_map2))
    cv.imwrite(EXPORT_FILENAME2, (height_map2 *255).astype(np.uint8))

    height_map3 = (height_map + height_map2)/2
    cv.imwrite(EXPORT_FILENAME3, (height_map3 *255).astype(np.uint8))

def normal_to_height_fourier(normal_map):
    # Perform Fourier Transform on the normal map
    f = np.fft.fft2(normal_map[:,:,0])  # Assuming normal_map[:,:,0] contains the X component of normal vectors
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Inverse Fourier Transform to get the height map
    height_map = np.fft.ifft2(fshift)
    height_map = np.abs(height_map)

    return height_map

def normal_to_height_fourier2(normal_map):
    height_map = np.zeros(normal_map.shape[:2])

    for i in range(3):  # Process each component separately (X, Y, Z)
        f = np.fft.fft2(normal_map[:,:,i])
        fshift = np.fft.fftshift(f)

        # Inverse Fourier Transform to get the height map for this component
        height_component = np.fft.ifft2(fshift)
        height_map += np.abs(height_component)  # Accumulate height information from all components

    return height_map

def DepthMap(surfNormals, maskImage):
    z = []

    print(surfNormals.shape)
    print(maskImage.shape)

    #maskImage = cv.cvtColor(maskImage, cv.COLOR_BGR2GRAY)

    nrows, ncols = maskImage.shape
    objectPixelRow, objectPixelCol = np.where(maskImage)
    objectPixels = np.column_stack((objectPixelRow, objectPixelCol))

    index = np.zeros((nrows, ncols))
    numPixels = objectPixels.shape[0]

    M = lil_matrix((2*numPixels, numPixels))
    b = np.zeros((2*numPixels, 1))

    for d in range(numPixels):
        pRow = objectPixels[d, 0]
        pCol = objectPixels[d, 1]
        nx = surfNormals[pRow, pCol, 0]
        ny = surfNormals[pRow, pCol, 1]
        nz = surfNormals[pRow, pCol, 2]

        if (index[pRow, pCol+1] > 0) and (index[pRow-1, pCol] > 0):
            M[2*d, index[pRow, pCol]] = 1
            M[2*d, index[pRow, pCol+1]] = -1
            b[2*d] = nx / nz

            M[2*d+1, index[pRow, pCol]] = 1
            M[2*d+1, index[pRow-1, pCol]] = -1
            b[2*d+1] = ny / nz

        # Other if-else conditions should be similarly translated

    x = np.linalg.lstsq(M.tocsr(), b, rcond=None)[0]
    x = x - np.min(x)

    tempShape = np.zeros((nrows, ncols))
    for d in range(numPixels):
        pRow = objectPixels[d, 0]
        pCol = objectPixels[d, 1]
        tempShape[pRow, pCol] = x[d]

    z = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            z[i, j] = tempShape[nrows-i-1, j]

    return z



def main(args):
    print("Running")

    # Run this first with calibration
    #calibrate_light_test(DATA_FOLDERNAME, MASK_FOLDERNAME, LIGHT_FOLDERNAME, 4, IMAGES_NAME, False) 
    rps = RPS()
    # Run this with the light file from the last to get normal map
    #runPStereo(rps)


    #runHeightGen(rps)
    #runHeightGen2()
    #compute_height_map()
    #compute_normal_map(rps)
    #compute_height_map2()
    #compute_my_heights()

    runTest()
    
    #runHeightModel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)