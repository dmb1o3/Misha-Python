from pickle import GLOBAL

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

# Stores clicked for SAM model
foreground_points = []
background_points = []
sam = sam_model_registry["vit_b"](checkpoint="./SAM Checkpoints/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
sam_image = None
masks = None

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
    cv2.imshow('Original Image', image)

    # Flat fielding

    # Image normalization
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # TESTING Display the normalized image
    cv2.imshow('Normalized Image', image)

    # Alignment
    
    # Denoising

    # Sampling patches

    # Rescaling
    image = cv2.resize(image, (0, 0), fx=downscale_factor, fy=downscale_factor)
    # TESTING Display the resized image
    cv2.imshow('Resized Image', image)

    # Wait for user to press key and then destroy all images
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image


def on_click(event):
    global foreground_points, background_points, masks

    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)

        if event.button == 1:  # Left-click (Foreground)
            foreground_points.append([x, y])
            print(f"Foreground Click: ({x}, {y})")

        elif event.button == 3:  # Right-click (Background)
            background_points.append([x, y])
            print(f"Background Click: ({x}, {y})")

        # Convert points to numpy arrays
        input_points = np.array(foreground_points + background_points)
        input_labels = np.array([1] * len(foreground_points) + [0] * len(background_points))

        # Get segmentation masks
        masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)

        # Display result
        plt.clf()
        plt.imshow(sam_image)

        if masks is not None:
            for mask in masks:
                # Create a semi-transparent white overlay
                overlay = np.zeros_like(sam_image, dtype=np.float32)
                overlay[:, :, :] = 255  # White color

                # Alpha blending (0.3 makes it semi-transparent)
                alpha = 0.3
                blended = np.where(mask[:, :, None], sam_image * (1 - alpha) + overlay * alpha, sam_image)

                plt.imshow(blended.astype(np.uint8))  # Show the blended result

        if foreground_points:
            plt.scatter(*zip(*foreground_points), color="red", marker="x", s=50, label="Foreground (Left-Click)")
        if background_points:
            plt.scatter(*zip(*background_points), color="blue", marker="o", s=50, label="Background (Right-Click)")

        plt.legend()
        plt.axis("off")
        plt.draw()

def on_close(event):
    global masks

    if masks is not None:
        mask = masks[0]  # Take the first mask
        segmented_image = sam_image.copy()
        segmented_image[~mask] = 0  # Set non-mask areas to black

        # Save the segmentation mask
        mask_path = "segmentation_mask.png"
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        # Save the segmented image
        segmented_path = "segmented_image.png"
        cv2.imwrite(segmented_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

        print(f"Mask saved at {mask_path}")
        print(f"Segmented image saved at {segmented_path}")


def generate_mask(image):
    global sam_image
    sam_image = image
    predictor.set_image(image)
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("close_event", on_close)
    plt.axis("off")
    plt.show()
    return


if __name__ == "__main__":
    generate_mask(cv2.imread("./2-10-25 Images/preprocessedImages/target/target_s.tiff"))