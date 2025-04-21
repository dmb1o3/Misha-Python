import cv2
import numpy as np

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


def preprocess_image(calibration, target, downscale_factor, directory, suffix):
    # Flat fielding
    #calibration = flat_field(calibration, flat)
    #target = flat_field(target, flat)

    # Image normalization
    calibration = cv2.normalize(calibration, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    target = cv2.normalize(target, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Rescaling
    calibration = cv2.resize(calibration, (0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_NEAREST)
    target = cv2.resize(target, (0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_NEAREST)

    # Save images
    cv2.imwrite(directory + "preprocessedImages/calibration_" + suffix, calibration)
    cv2.imwrite(directory + "preprocessedImages/target/target_" + suffix, target)

