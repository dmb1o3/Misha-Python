import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from calibrateLights import calibrate_light
from downcaledTargetNormals import generate_normal_map
from SAMSegmenter import SAMSegmenter, checkpoints
from preprocess import preprocess_image
from matlab2python import depth_map
from matplotlib import cm
from height_map import estimate_height_map
import os
import cv2
import plotly.graph_objects as go

NORMAL_MAP_A_PATH: str = (
    "https://raw.githubusercontent.com/YertleTurtleGit/depth-from-normals/main/normal_mapping_a.png"
)
NORMAL_MAP_B_PATH: str = (
    "https://raw.githubusercontent.com/YertleTurtleGit/depth-from-normals/main/normal_mapping_b.png"
)

BLENDER_MAP: str = (
    "target_e_normals.png"
)

NORMAL_MAP_A_IMAGE: np.ndarray = io.imread(NORMAL_MAP_A_PATH)
NORMAL_MAP_B_IMAGE: np.ndarray = io.imread(NORMAL_MAP_B_PATH)
NORMAL_MAP_BLENDER_IMAGE: np.ndarray = io.imread(BLENDER_MAP)

def check_make_folder(directory):
    """
    Will make sure that a folder exists if not will create it. Checks to make sure
    previous folders are generated as well
    """
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
    image_data = {}
    # Read in all images in the directory, preprocess them and then save them in a folder in directory
    for filename in os.listdir(directory):
        # Try to read image and if not none we have a valid image
        img = cv2.imread(os.path.join(directory,filename))
        if img is not None:
            prefix = filename.split("_")
            suffix = prefix[1]
            prefix = prefix[0]
            # Check to see if we have subdict set up. If not set it up to store image
            if suffix not in image_data:
                image_data[suffix] = {"target":None, "calibration":None, "flat":None}

            image_data[suffix][prefix] = img

    for key in image_data:
        preprocess_image(image_data[key]["calibration"], image_data[key]["target"], image_data[key]["flat"], 0.5, directory, key)
        calibration_mask = "calibration_" + key
        target_mask = "target_" + key

    print(calibration_mask)
    print(target_mask)

    # Select a calibration and target frame and generate a mask for each of them
    segmenter = SAMSegmenter("vit_b", checkpoints["vit_b"])
    segmenter.generate_mask(
        directory + "preprocessedImages/" + calibration_mask,
        directory + "preprocessedImages/mask/calibration"
    )
    segmenter.generate_mask(
        directory + "preprocessedImages/target/" + target_mask,
        directory + "preprocessedImages/mask/target"
    )


def run():
    directory = "./3-5-2025/"
    # Preprocess images and generate old_mask for calibration and target
    #preprocess_generate_mask(directory)
    # Update the directory to now use the preprocessed images
    directory = directory + "preprocessedImages/"
    # Calibrate the lights
    #calibrate_light(directory, 4)
    # Generate the normal maps
    normals = generate_normal_map(directory)
    z = depth_map(normals, cv2.imread(directory + "mask/target_mask.tiff", cv2.IMREAD_GRAYSCALE))
    # To display the surface normals
    fig = go.Figure(data=[go.Surface(z=z, colorscale='gray')])
    fig.update_layout(title='Depth Map', autosize=False, width=800, height=800, margin=dict(l=65, r=50, b=65, t=90))
    fig.show()





if __name__ == "__main__":
    run()