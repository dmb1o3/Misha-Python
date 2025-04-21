import cv2
import numpy as np
from matplotlib import pyplot as plt

def flat_field(image, flat):
    """
    typ = img.dtype
    img = img.astype(np.float32)
    flat = flat.astype(np.float32)

    corrected = img * flat

    # Clip values to valid range and convert back to original type
    corrected = np.clip(corrected, 0, 255).astype(typ)
    """
    original_mean = np.mean(image)

    # Apply correction
    corrected = image.astype(np.float32) * flat.astype(np.float32)

    # Scale to maintain original average brightness
    corrected = corrected * (original_mean / np.mean(corrected))


    corrected = np.clip(corrected, 0, 255).astype(image.dtype)
    return corrected


def flat_field2(image, flat):
    """
    typ = img.dtype
    img = img.astype(np.float32)
    flat = flat.astype(np.float32)

    corrected = img * flat

    # Clip values to valid range and convert back to original type
    corrected = np.clip(corrected, 0, 255).astype(typ)
    """
    image = image.astype(np.float32)
    flat_field = flat.astype(np.float32)

    # Normalize the flat field image
    flat_field_normalized = flat_field / np.mean(flat_field)
    
    # Perform flat field correction
    corrected_image = image / flat_field_normalized
    
    # Clip the values to ensure they are within the valid range
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    return corrected_image


def preprocess_image(calibration, target, flat, downscale_factor, directory, suffix):
    #figure, axes = plt.subplots(4, 2)
    #axes[0][0].set_title("Pics & fields")
    #_ = axes[0][0].imshow(target, cmap="RdBu")
    #_ = axes[1][0].imshow(flat, cmap="RdBu")

    # Flat fielding
    calibration = flat_field2(calibration, flat)
    target = flat_field2(target, flat)

    #_ = axes[2][0].imshow(target, cmap="RdBu")
    #plt.show()

    # Image normalization
    calibration = cv2.normalize(calibration, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    target = cv2.normalize(target, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Rescaling
    calibration = cv2.resize(calibration, (0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_NEAREST)
    target = cv2.resize(target, (0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_NEAREST)

    #_ = axes[3][0].imshow(target, cmap="RdBu")
    #plt.show()

    # Save images
    cv2.imwrite(directory + "preprocessedImages/calibration_" + suffix, calibration)
    cv2.imwrite(directory + "preprocessedImages/target/target_" + suffix, target)

