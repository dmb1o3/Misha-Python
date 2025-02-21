from calibrateLights import calibrate_light
from downcaledTargetNormals import generate_normal_map
from SAMSegmenter import SAMSegmenter, checkpoints


def preprocess_generate_mask(directory, directions):
    # Read in all images in the directory

    # Preprocess the images

    # Save the images in the directory in folder preprocessImages

    # Select a calibration and target frame and generate a mask for each of them
    segmenter = SAMSegmenter("vit_l", checkpoints["vit_l"])
    segmenter.generate_mask(
        "./2-10-25 Images/preprocessedImages/calibration_s.tiff",
        "./2-10-25 Images/preprocessedImages/mask/calibration"
    )
    segmenter.generate_mask(
        "./2-10-25 Images/preprocessedImages/target/target_s.tiff",
        "./2-10-25 Images/preprocessedImages/mask/target"
    )


    return


def run():
    directory = "./2-10-25 Images/preprocessedImages/"
    directions = ["n", "e", "s", "w"]
    # Preprocess images and generate old_mask for calibration and target
    preprocess_generate_mask(directory, directions)

    # Calibrate the lights
    calibrate_light(directory, 4)
    # Generate the normal maps
    generate_normal_map(directory)



    # Generate depth map


if __name__ == "__main__":
    run()