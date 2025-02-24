import cv2

def flat_field(image):
    # Get image resolution

    # Create a flat field image of uniform distribution of same resolution

    # Apply flat field image

    return image


def preprocess_image(image, downscale_factor):
    """

    :param image:
    :param downscale_factor:
    :return:
    """
    # TESTING show the original image
    #cv2.imshow('Original Image', image)

    # Flat fielding

    # Image normalization
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # TESTING Display the normalized image
    #cv2.imshow('Normalized Image', image)

    # Alignment

    # Denoising

    # Sampling patches

    # Rescaling
    image = cv2.resize(image, (0, 0), fx=downscale_factor, fy=downscale_factor)
    # TESTING Display the resized image
    #cv2.imshow('Resized Image', image)

    # Wait for user to press key and then destroy all images
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return image
