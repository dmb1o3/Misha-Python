import cv2
import numpy as np
import os


def calibrate_light(directory: str, num_lights: int) -> np.ndarray:
    """
    Calibrate lighting direction using chrome sphere images.

    Args:
        directory (str): Path to directory containing chrome sphere images
        num_lights (int): Number of light sources to calibrate

    Returns:
        np.ndarray: Array of light directions, shape (num_lights, 3)
    """
    # Ensure directory path ends with separator
    if not directory.endswith('/'):
        directory += '/'

    # Read the chrome sphere mask
    mask_filename = os.path.join(directory, 'calibration_mask.tiff')
    circle = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

    # Calculate chrome sphere center and radius
    max_val = np.max(circle)
    circle_coords = np.where(circle == max_val)
    circle_rows, circle_cols = circle_coords[0], circle_coords[1]

    # Calculate sphere parameters
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
    calibration_names = ["n", "e", "s", "w"]
    # Process each light source
    for i in range(num_lights):
        # Read chrome sphere image for current light
        img_filename = os.path.join(directory, f'calibration_{calibration_names[i]}.tiff')
        image = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)

        # Find brightest point
        max_val = np.max(image)
        point_coords = np.where(image == max_val)
        point_rows, point_cols = point_coords[0], point_coords[1]

        # Calculate average position of brightest points
        px = np.mean(point_cols)
        py = np.mean(point_rows)

        # Calculate surface normal at brightest point
        Nx = px - xc
        Ny = -(py - yc)  # Note the negative sign to match MATLAB convention
        Nz = np.sqrt(radius ** 2 - Nx ** 2 - Ny ** 2)

        # Normalize the normal vector
        normal = np.array([Nx, Ny, Nz]) / radius

        # Calculate light direction using reflection formula
        NR = np.dot(normal, R)
        L[i] = 2 * NR * normal - R

    # Save calibrated light directions
    output_file = os.path.join(directory, 'calibrated_light.txt')
    with open(output_file, 'w') as f:
        f.write(f"{num_lights}\n")
        for light_dir in L:
            f.write(f" {light_dir[0]:10.5f} {light_dir[1]:10.5f} {light_dir[2]:10.5f}\n")

    return L


# Example usage:
if __name__ == "__main__":
    light_directions = calibrate_light("./preprocessedImages/calibrate", 4)
    print("Calibrated light directions:")
    print(light_directions)