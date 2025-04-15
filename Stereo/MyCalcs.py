from __future__ import print_function
from rps import RPS
from calibrate_light import calibrate_light_test

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
from matplotlib.colors import TwoSlopeNorm
from typing import List, Tuple
from scipy.signal import convolve2d

#NORMAL_MAP_A_PATH: str = ('./data/results/shapes_normal.png')
NORMAL_MAP_A_PATH: str = ('./data/results/innard_normal.jpg')
#NORMAL_MAP_A_PATH: str = ('./data/results/monkey_normal.png')
#NORMAL_MAP_A_PATH: str = ('./data/results/target_normal.png')
#NORMAL_MAP_A_PATH: str = ('./data/results/bricks.jpg')
EXPORT_FILENAME = './data/results/export_normal.png'
EXPORT_FILENAME2 = './data/results/export_normal2.png'
EXPORT_FILENAME3 = './data/results/export_normal3.png'

NORMAL_MAP_A_IMAGE: np.ndarray = io.imread(NORMAL_MAP_A_PATH)

scale = 1.0

def calculate_gradients(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normals = normals.astype(np.float64)

    horizontal_angle_map = np.arccos(normals[..., 0])
    left_gradients = np.sign(horizontal_angle_map - np.pi / 2) * (
        1 - np.sin(horizontal_angle_map)
    )

    vertical_angle_map = np.arccos(np.clip(normals[..., 1], -1, 1))
    top_gradients = -np.sign(vertical_angle_map - np.pi / 2) * (
        1 - np.sin(vertical_angle_map)
    )

    return left_gradients, top_gradients


def show_gradients():
    normals = ((NORMAL_MAP_A_IMAGE[:, :, :3] / 255) - 0.5) * 2
    print(normals.shape)
    left_gradients, top_gradients = calculate_gradients(normals)


    figsize = (14, 14)
    figure, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].set_title("anisotropic left gradients (left to right)")
    _ = axes[0].imshow(left_gradients, cmap="RdBu", norm=TwoSlopeNorm(0))
    axes[1].set_title("anisotropic top gradients (top to bottom)")
    _ = axes[1].imshow(top_gradients, cmap="RdBu", norm=TwoSlopeNorm(0))
    axes[2].set_title("normals (clipped)")
    _ = axes[2].imshow(np.clip(normals, 0, 255))
    print(NORMAL_MAP_A_IMAGE.shape)

    plt.show()
    
    #convlr = (left_gradients-np.min(left_gradients))/(np.max(left_gradients)-np.min(left_gradients))
    #cv.imwrite(EXPORT_FILENAME2, (convlr *255).astype(np.uint8))
    #cv.imwrite(EXPORT_FILENAME3, (convy *255).astype(np.uint8))


def apply_kernels_to_gradients2(gradient_x, gradient_y):
    #print(gradient_x[0])
    print(gradient_x.shape)
    max_value = np.max(gradient_x)
    min_value = np.min(gradient_x)
    print(f"Highest gradient value: {max_value}")
    print(f"Lowest gradient value: {min_value}")
    figsize = (14, 14)
    figure, axes = plt.subplots(2, 3, figsize=figsize)
    axes[0][0].set_title("left gradients (left to right)")
    _ = axes[0][0].imshow(gradient_x, cmap="RdBu", norm=TwoSlopeNorm(0))

    _ = axes[1][0].imshow(np.abs(gradient_x), cmap="RdBu", norm=TwoSlopeNorm(0))

    norm = TwoSlopeNorm(0)
    equilized_gradient = norm(gradient_x)

    equilized_gradient_y = norm(gradient_y)
    #_ = axes[1][2].imshow(equilized_gradient, cmap="RdBu")

    #x_gradient = test
    #x_gradient = transform.resize(x_gradient, (int(x_gradient.shape[0] * scale), int(x_gradient.shape[1] * scale)), anti_aliasing=True)
    #x_gradient = (x_gradient-np.min(x_gradient))/(np.max(x_gradient)-np.min(x_gradient))
    #x_gradient = (x_gradient * 255).astype(np.uint8)
    #print(x_gradient[0])
    #cv.imshow("gradient", x_gradient)
    #cv.waitKey(0)

    height_map = np.zeros_like(equilized_gradient)

    # Define kernels for X, Y, and Z directions
    kernel_x = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    #kernel_y = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    #kernel_z = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

    # Apply kernels to the gradients in each direction
    height_map += np.abs(convolve2d(equilized_gradient, kernel_x, mode='same'))
    #height_map += np.abs(convolve2d(gradient_y, kernel_y, mode='same'))
    #height_map += np.abs(convolve2d(gradient_z, kernel_z, mode='same'))

    _ = axes[0][1].imshow(height_map)

    height_map = (height_map-np.min(height_map))/(np.max(height_map)-np.min(height_map))
    height_map = (height_map * 255).astype(np.uint8)

    print(f"Highest value: {np.max(height_map)}")
    print(f"Lowest value: {np.min(height_map)}")

    cv.imshow("height", height_map)
    cv.waitKey(0)

    #print(height_map[0])

    comb = np.ones((height_map.shape[0], height_map.shape[1])) * 127
    #print(normal_map[0])
    max_val = np.max(height_map)
    min_val = np.min(height_map)
    #normed = (height_map - 0) / (max_val - 0)
    normed = (((height_map - 0) / (1 - -1)) * (max_val - min_val) + 0) 

    print(f"Highest value: {np.max(normed)}")
    print(f"Lowest value: {np.min(normed)}")
    for row in range(height_map.shape[0]):  # Iterate over rows
        row_val = 0
        for col in range(height_map.shape[1]):
            row_val += normed[row][col] #/2.0 - 127
            comb[row][col] += row_val
    _ = axes[0][2].imshow(comb, cmap="RdBu", norm=TwoSlopeNorm(0))


    #normed2 = (((equilized_gradient - 0) / (1 - -1)) * (max_val - min_val) + 0) 
    #normed2 = (normed2 * 255).astype(np.int8)
    normed2 = equilized_gradient
    normedY = equilized_gradient_y
    print(f"Highest normed Y value: {np.max(normedY)}")
    print(f"Lowest normed Y value2: {np.min(normedY)}")
    print(normed2[0][:10])
    comb2 = np.zeros((equilized_gradient.shape[0], equilized_gradient.shape[1])) #* 127
    combY = np.zeros((equilized_gradient_y.shape[0], equilized_gradient_y.shape[1])) #* 127
    #comb3 = np.zeros((equilized_gradient.shape[0], equilized_gradient.shape[1])) #* 127
    #reverse_image = np.fliplr(normed2)
    #comb3 = normed2/255#(normed2 + reverse_image)/2/255
    for row in range(equilized_gradient.shape[0]):  # Iterate over rows
        row_val = 0
        for col in range(equilized_gradient.shape[1]):
            #comb3[row][col] = ((normed2[row][col] + reverse_image[row][col])/255)/2
            #row_val += (normed2[row][col] -127)/ 128
            row_val += (normed2[row][col] -0.5)
            #row_val = np.add(row_val, normed2[row][col]  - 127, dtype=np.float64)
            #print(row_val)
            comb2[row][col] += row_val 
            #if row == 100 and ((col > 170 and col < 180) or (col > 320 and col < 330)):
             #   print(f"Row: {row}, Col: {col} = Normed: {normed2[row][col]}")
              #  print(f"Row: {row}, Col: {col} = Val: {row_val}")
              #  print(f"Row: {row}, Col: {col} = Val: {comb2[row][col]}")
        cv.waitKey(0)

    for col in range(equilized_gradient_y.shape[1]):  # Iterate over rows
        col_val = 0
        for row in range(equilized_gradient_y.shape[0]):
            #comb3[row][col] = ((normed2[row][col] + reverse_image[row][col])/255)/2
            #row_val += (normed2[row][col] -127)/ 128
            col_val += (normedY[row][col] -0.5)
            #row_val = np.add(row_val, normed2[row][col]  - 127, dtype=np.float64)
            #print(row_val)
            combY[row][col] += col_val 
        cv.waitKey(0)
    print(comb2[0][:10])
    _ = axes[1][1].imshow(comb2, cmap="RdBu")
    _ = axes[1][2].imshow(combY, cmap="RdBu")
    #_ = axes[1][2].imshow(comb3, cmap="RdBu")

    plt.show()

    return height_map


def apply_kernels_to_gradients(gradient_x, gradient_y):
    print(gradient_x.shape)
    max_value_x = np.max(gradient_x)
    min_value_x = np.min(gradient_x)
    print(f"Highest x gradient value: {max_value_x}")
    print(f"Lowest x gradient value: {min_value_x}")

    max_value_y = np.max(gradient_y)
    min_value_y = np.min(gradient_y)
    print(f"Highest y gradient value: {max_value_y}")
    print(f"Lowest y gradient value: {min_value_y}")

    figsize = (14, 14)
    figure, axes = plt.subplots(4, 3, figsize=figsize)
    axes[0][0].set_title("left gradients (left to right)")
    _ = axes[0][0].imshow(gradient_x, cmap="RdBu", norm=TwoSlopeNorm(0))
    _ = axes[0][1].imshow(gradient_y, cmap="RdBu", norm=TwoSlopeNorm(0))

    _ = axes[1][0].imshow(np.abs(gradient_x), cmap="RdBu", norm=TwoSlopeNorm(0))
    _ = axes[1][1].imshow(np.abs(gradient_y), cmap="RdBu", norm=TwoSlopeNorm(0))

    norm = TwoSlopeNorm(0)
    equilized_gradient_x = norm(gradient_x)
    norm = TwoSlopeNorm(0)
    equilized_gradient_y = norm(gradient_y)
    #max_val = np.max(gradient_x)
    #min_val = np.min(gradient_x)
    #quilized_gradient_x = (((gradient_x - 0) / (1 - -1)) * (max_val - min_val) + 0) *2

    normedX = equilized_gradient_x
    normedY = equilized_gradient_y

    print(f"Highest normed X value: {np.max(normedX)}")
    print(f"Lowest normed X value2: {np.min(normedX)}")
    print(f"Highest normed Y value: {np.max(normedY)}")
    print(f"Lowest normed Y value2: {np.min(normedY)}")

    combX = np.zeros((equilized_gradient_x.shape[0], equilized_gradient_x.shape[1])) #* 127
    combY = np.zeros((equilized_gradient_y.shape[0], equilized_gradient_y.shape[1])) #* 127
    combX2 = np.zeros((equilized_gradient_x.shape[0], equilized_gradient_x.shape[1])) #* 127
    combY2 = np.zeros((equilized_gradient_y.shape[0], equilized_gradient_y.shape[1])) #* 127

    for row in range(equilized_gradient_x.shape[0]):  # Iterate over rows
        row_val = 0
        row_val_neg = 0
        for col in range(equilized_gradient_x.shape[1]):
            row_val += (normedX[row][col] -0.5)
            combX[row][col] += row_val 
            row_val_neg += (normedX[row][equilized_gradient_x.shape[1] - col -1] -0.5)
            combX2[row][equilized_gradient_x.shape[1] - col -1] += row_val_neg 
            if row == 100 and ((col > 170 and col < 180) or (col > 320 and col < 330)):
                print(f"Row: {row}, Col: {col} = Normed: {normedX[row][col]}")
                print(f"Row: {row}, Col: {col} = Val: {row_val}")
                print(f"Row: {row}, Col: {col} = Val: {combX[row][col]}")
            if row == 5 and ((col > 170 and col < 180) or (col > 320 and col < 330)):
                print(f"Row: {row}, Col: {col} = Normed: {normedX[row][col]}")
                print(f"Row: {row}, Col: {col} = Val: {row_val}")
                print(f"Row: {row}, Col: {col} = Val: {combX[row][col]}")
        cv.waitKey(0)

    for col in range(equilized_gradient_y.shape[1]):  # Iterate over rows
        col_val = 0
        col_val_neg = 0
        for row in range(equilized_gradient_y.shape[0]):
            col_val += (normedY[row][col] -0.5)
            combY[row][col] += col_val 
            col_val_neg += (normedY[equilized_gradient_y.shape[0] - row -1][col] -0.5)
            combY2[equilized_gradient_y.shape[0] - row -1][col] += col_val_neg 
        cv.waitKey(0)
    print(combX[0][:10])

    _ = axes[2][0].imshow(combX, cmap="RdBu", norm=TwoSlopeNorm(0))
    _ = axes[2][1].imshow(combY, cmap="RdBu", norm=TwoSlopeNorm(0))
    _ = axes[3][0].imshow(combX2, cmap="RdBu", norm=TwoSlopeNorm(0))
    _ = axes[3][1].imshow(combY2, cmap="RdBu", norm=TwoSlopeNorm(0))

    norm = TwoSlopeNorm(0)
    temp_r = norm(combX)
    norm = TwoSlopeNorm(0)
    temp_l = norm(combX2)

    norm = TwoSlopeNorm(0)
    temp_yr = norm(combY)
    norm = TwoSlopeNorm(0)
    temp_yl = norm(combY2)

    _ = axes[0][2].imshow((temp_r - temp_l), cmap="RdBu")
    _ = axes[1][2].imshow((temp_yr - temp_yl), cmap="RdBu")
    #_ = axes[3][1].imshow(combY2, cmap="RdBu")

    plt.show()

    return 0



def old_code(gradient_x):

    max_val = np.max(gradient_x)
    min_val = np.min(gradient_x)

    negative_indices = np.where((gradient_x * 255).astype(np.int8) <= 0)
    positive_indices = np.where((gradient_x * 255).astype(np.int8) >= 0) #np.where(gradient_x > 0)[0]

    #print(f"Pos Shape: {positive_indices.shape}, Neg Shape: {negative_indices.shape}, Total: {positive_indices.shape[0] + negative_indices.shape[0]} which should be: {gradient_x.shape[0] * gradient_x.shape[1]}")

    negative_nums = np.zeros_like(gradient_x)
    positive_nums = np.zeros_like(gradient_x)
    negative_nums[negative_indices] = gradient_x[negative_indices] * -1
    positive_nums[positive_indices] = gradient_x[positive_indices]

    print(negative_nums.shape)
    print(positive_nums.shape)

    neg_normed = (negative_nums-np.min(negative_nums))/(np.max(negative_nums)-np.min(negative_nums)) #(((negative_nums - 0) / (1 - -1)) * (np.max(negative_nums) - np.min(negative_nums)) + 0) 
    pos_normed = (positive_nums-np.min(positive_nums))/(np.max(positive_nums)-np.min(positive_nums)) #(((positive_nums - 0) / (1 - -1)) * (np.max(positive_nums) - np.min(positive_nums)) + 0) 

    print(f"Highest neg attempt value: {np.max(neg_normed)}")
    print(f"Lowest neg attempt value: {np.min(neg_normed)}")
    print(f"Highest pos attempt value: {np.max(pos_normed)}")
    print(f"Lowest pos attempt value: {np.min(pos_normed)}")

    # Rejoin the negative and positive numbers while maintaining the original order
    rejoined_data = np.empty_like(gradient_x)
    rejoined_data[negative_indices] = (neg_normed[negative_indices] * 1)
    print(f"Lowest attempt A value: {np.min(neg_normed)}")
    print(f"Lowest attempt B value: {np.min(rejoined_data)}")
    rejoined_data[positive_indices] = pos_normed[positive_indices]
    print(f"Lowest attempt C value: {np.min(rejoined_data)}")
    min_index = np.unravel_index(np.argmin(rejoined_data), rejoined_data.shape)
    #for value in min_index:
        #print(f"Index: {value}")
    #print(neg_normed.shape)

    print(f"Highest attempt value: {np.max(rejoined_data)}")
    print(f"Lowest attempt value: {np.min(rejoined_data)}")

    _ = axes[1][1].imshow(rejoined_data, cmap="RdBu", norm=TwoSlopeNorm(0))

def compute_my_heights():
    normal_map: np.ndarray = io.imread(NORMAL_MAP_A_PATH)
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

def compute_x_heights(x_gradient):

    x_gradient = transform.resize(x_gradient, (int(x_gradient.shape[0] * scale), int(x_gradient.shape[1] * scale)), anti_aliasing=True)
    x_gradient = (x_gradient-np.min(x_gradient))/(np.max(x_gradient)-np.min(x_gradient))
    x_gradient = (x_gradient * 255).astype(np.uint8)
    #x_gradient = (x_gradient * 255).astype(int)
    print(x_gradient[0])
    cv.imshow("test", x_gradient)
    cv.waitKey(0)
    height_map = np.ones((x_gradient.shape[0], x_gradient.shape[1])) * 127
    for row in range(x_gradient.shape[0]):  # Iterate over rows
        row_val = 0
        for col in range(x_gradient.shape[1]):
            row_val += x_gradient[row][col]/2.0 - 64
            height_map[row][col] += row_val
        #print(map[row])
        #print(height_map[row])
    #print(height_map[0])
    height_map = (height_map-np.min(height_map))/(np.max(height_map)-np.min(height_map))
    #print(height_map[20])
    cv.imwrite(EXPORT_FILENAME, (height_map *255).astype(np.uint8))
    #cv.waitKey(0)


def run_this():
    normals = ((NORMAL_MAP_A_IMAGE[:, :, :3] / 255) - 0.5) * 2
    show_gradients()
    left_gradients, top_gradients = calculate_gradients(normals)

    #compute_x_heights(left_gradients)

    map = apply_kernels_to_gradients(left_gradients, top_gradients)
    #map = left_gradients
    #convlr = (map-np.min(map))/(np.max(map)-np.min(map))
    #print(convlr[0])
    #cv.imwrite(EXPORT_FILENAME2, (convlr *255).astype(np.uint8))

def runTest():
    show_gradients()

def main(args):
    run_this()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
