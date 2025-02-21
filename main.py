from calibrateLights import calibrate_light
from downcaledTargetNormals import generate_normal_map
from preprocess import preprocess_image, generate_mask


def preprocess_generate_mask(directory):
    # Read in all images in the directory

    # Preprocess the images

    # Save the images in the directory in folder preprocessImages

    # Select a calibration and target frame and generate a mask for each of them

    return


def run():
    directory = "./2-10-25 Images/preprocessedImages/"
    # Calibrate the lights
    calibrate_light(directory, 4)
    # Generate the normal maps
    generate_normal_map(directory)



    # Generate depth map


if __name__ == "__main__":
    run()