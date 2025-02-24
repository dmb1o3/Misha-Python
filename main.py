from calibrateLights import calibrate_light
from downcaledTargetNormals import generate_normal_map
from SAMSegmenter import SAMSegmenter, checkpoints
from MSDTest import generate_depth_map
from preprocess import preprocess_image
import os
import cv2

def check_make_folder(directory):
    os.makedirs(directory, exist_ok=True)


def load_calibration_target_images(folder):
    calibration_images = []
    target_images = []
    # Loop through all files in directory
    for filename in os.listdir(folder):
        # Try to read image and if not none we have a valid image
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            # Check if image is for calibration or for target
            if filename.split("_")[0] == "calibration":
                calibration_images.append(img)
            else:
                target_images.append(img)
    return calibration_images, target_images


def preprocess_generate_mask(directory):
    # Make sure folder we want to use to store preproceesed images exists. If not create
    check_make_folder(directory + "preprocessedImages/target/")
    check_make_folder(directory + "preprocessedImages/mask/")
    # Used to save the last calibration and target file names preprocessed to then make a mask
    calibration_mask = ""
    target_mask = ""
    # Read in all images in the directory, preprocess them and then save them in a folder in directory
    for filename in os.listdir(directory):
        # Try to read image and if not none we have a valid image
        img = cv2.imread(os.path.join(directory,filename))
        if img is not None:
            # Preprocess images
            img = preprocess_image(img, 0.5)
            # Check if image is for calibration or for target
            if filename.split("_")[0] == "calibration":
                # Save calibration image
                cv2.imwrite(directory + "/preprocessedImages/" + filename, img)
                calibration_mask = filename

            else:
                # Save target image
                cv2.imwrite(directory + "/preprocessedImages/target/" + filename, img)
                target_mask = filename


    # Select a calibration and target frame and generate a mask for each of them
    segmenter = SAMSegmenter("vit_l", checkpoints["vit_l"])
    segmenter.generate_mask(
        directory + "preprocessedImages/" + calibration_mask,
        directory + "preprocessedImages/mask/calibration"
    )
    segmenter.generate_mask(
        directory + "preprocessedImages/target/" + target_mask,
        directory + "preprocessedImages/mask/target"
    )


    return


def run():
    directory = "./TestImages/"
    # Preprocess images and generate old_mask for calibration and target
    preprocess_generate_mask(directory)
    # Update the directory to now use the preprocessed images
    directory = directory + "preprocessedImages/"
    # Calibrate the lights
    calibrate_light(directory, 4)
    # Generate the normal maps
    generate_normal_map(directory)


if __name__ == "__main__":
    run()