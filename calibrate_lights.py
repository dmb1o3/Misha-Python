import os
import cv2
import numpy as np


def apply_mask(image, mask, xc, yc, radius):
    rows, cols = image.shape

    # Apply the Mask
    image[mask == 0] = 0  # Set all pixels outside the mask to black

    # Keep Only the Brightest Spots
    max_val = np.max(image)  # Find the brightest value
    image[image < max_val] = 0  # Set all pixels not at max brightness to black

    # Trim Excess Pixels
    y_indices, x_indices = np.indices((rows, cols))  # Create grid of coordinates
    distances = np.sqrt((y_indices - yc) ** 2 + (x_indices - xc) ** 2)  # Compute distances from center
    image[distances >= (radius - 5)] = 0  # Set pixels outside the defined radius to black

    return image


def calibrate_light(directory, num_lights):
    """
    Given a directory with calibration photos and the number of lights will caculate light vectors and store in
    a text file in directory/calibrate_light.txt

    :param directory: Directory with calibration images of metal ball. Must be only photos in the directory
    :param num_lights: The number of light directions
    :return: Does not return anything
    """

    # Get the mask
    mask_filename = os.path.join(directory, 'mask/calibration_mask.tiff')
    circle = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

    # Calculate chrome sphere center and radius
    max_val = np.max(circle)
    circle_coords = np.where(circle == max_val)
    circle_rows, circle_cols = circle_coords[0], circle_coords[1]
    max_row, min_row = np.max(circle_rows), np.min(circle_rows)
    max_col, min_col = np.max(circle_cols), np.min(circle_cols)
    xc = (max_col + min_col) / 2
    yc = (max_row + min_row) / 2
    radius = (max_row - min_row) / 2
    print(f"Center: ({xc}, {yc})")
    print(f"Radius: {radius}")

    # Initialize variables
    R = np.array([0, 0, 1.0])  # Reflection direction
    L = np.zeros((num_lights, 3))  # Light directions matrix
    i = 0
    # Process each light source
    for filename in os.listdir(directory):
        image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Apply the mask
            masked_image = apply_mask(image, circle, xc, yc, radius)

            # Find the brightest point
            max_val = np.max(masked_image)
            point_cords = np.where(image == max_val)
            point_rows, point_cols = point_cords[0], point_cords[1]

            # Calculate average position of the brightest points
            px = np.mean(point_cols)
            py = np.mean(point_rows)

            # Calculate surface normal at brightest point
            Nx = px - xc
            Ny = -(py - yc)
            Nz = np.sqrt(radius ** 2 - Nx ** 2 - Ny ** 2)

            # Normalize the normal vector
            normal = np.array([Nx, Ny, Nz]) / radius

            # Calculate light direction using reflection formula
            NR = np.dot(normal, R)
            L[i] = 2 * NR * normal - R
            i += 1

    # Save calibrated light directions
    output_file = os.path.join(directory, 'calibrated_light.txt')
    with open(output_file, 'w') as f:
        for light_dir in L:
            f.write(f"{light_dir[0]:.5f} {light_dir[1]:.5f} {light_dir[2]:.5f}\n")

