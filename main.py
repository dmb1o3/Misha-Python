import os
import cv2
import plotly.graph_objects as go
from tools.preprocess import preprocess_image
from tools.depth_map import generate_depth_map
from tools.SAMSegmenter import SAMSegmenter, checkpoints
from tools.normal_map import generate_normal_map
from tools.calibrate_lights import calibrate_light

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
    check_make_folder(directory + "preprocessedImages/normals/")
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
                image_data[suffix] = {"target":None, "calibration":None} #, "flat":None}

            image_data[suffix][prefix] = img

    for key in image_data:
        preprocess_image(image_data[key]["calibration"], image_data[key]["target"], 0.5, directory, key)
        calibration_mask = "calibration_" + key
        target_mask = "target_" + key

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


def run():
    # Set directory with images
    directory = "./3-19-2025/"
    # Preprocess images and generate mask for calibration and target
    preprocess_generate_mask(directory)
    # Update the directory to now use the preprocessed images we just generated
    directory = directory + "preprocessedImages/"
    # Calibrate the lights
    calibrate_light(directory, 8)
    # Generate the normal maps
    generate_normal_map(directory)
    # Generate the depth map
    depth_map, corrected_depth_map, fitted_curve = generate_depth_map(directory + "normals/normal_map.npy", directory + "depth_map")
    # Create figure
    fig = go.Figure()
    # Add original depth map
    fig.add_trace(go.Surface(z=depth_map, colorscale='gray', opacity=0.7, name="Original Depth"))
    # Add fitted polynomial surface
    fig.add_trace(go.Surface(z=fitted_curve, colorscale='viridis', opacity=0.6, name="Fitted Curve"))
    # Update layout
    fig.update_layout(title='Original Depth Map vs Fitted Polynomial Surface', autosize=True)
    fig.show()
    # Display the corrected depth map
    fig2 = go.Figure()
    fig2 = go.Figure(data=[go.Surface(z=corrected_depth_map, colorscale='gray')])
    #fig2.update_layout(title='Corrected Depth Map', autosize=True)
    fig2.update_layout(title='Corrected Depth Map',
       scene=dict(zaxis=dict(range=[-40, 2000])),autosize=True)
    fig2.show()
    # Display Original depth map
    fig3 = go.Figure()
    fig3 = go.Figure(data=[go.Surface(z=depth_map, colorscale='gray')])
    fig3.update_layout(title='Depth Map', autosize=True)
    fig3.show()


if __name__ == "__main__":
    run()
