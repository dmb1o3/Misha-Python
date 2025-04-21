import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

checkpoints = {"vit_b":"./SAM Checkpoints/sam_vit_b_01ec64.pth",
               "vit_h":"./SAM Checkpoints/sam_vit_h_4b8939.pth",
               "vit_l":"./SAM Checkpoints/sam_vit_l_0b3195.pth"}

class SAMSegmenter:
    def __init__(self, model_type, model_path):
        # Initialize SAM model
        self.sam = sam_model_registry[model_type](checkpoint=model_path)
        self.predictor = SamPredictor(self.sam)

        # State variables
        self.foreground_points = []
        self.background_points = []
        self.sam_image = None
        self.masks = None
        self.file_destination = None
        self.current_figure = None

    def load_image(self, image_path):
        """Load and prepare image for SAM"""
        try:
            # Load image and convert to RGB
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Store image and set in predictor
            self.sam_image = image
            self.predictor.set_image(image)
            return True
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return False

    def reset_points(self):
        """Reset all points and masks"""
        self.foreground_points = []
        self.background_points = []
        self.masks = None

    def _on_click(self, event):
        """Handle click events for point selection"""
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Add points based on click type
        if event.button == 1:  # Left-click (Foreground)
            self.foreground_points.append([x, y])
            print(f"Added foreground point: ({x}, {y})")
        elif event.button == 3:  # Right-click (Background)
            self.background_points.append([x, y])
            print(f"Added background point: ({x}, {y})")

        self._update_mask()
        self._update_display()

    def _update_mask(self):
        """Update segmentation mask based on current points"""
        if not (self.foreground_points or self.background_points):
            return

        try:
            # Prepare points and labels
            input_points = np.array(self.foreground_points + self.background_points)
            input_labels = np.array([1] * len(self.foreground_points) + [0] * len(self.background_points))

            # Get new masks
            self.masks, _, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False
            )
        except Exception as e:
            print(f"Error updating mask: {str(e)}")

    def _update_display(self):
        """Update the display with current mask and points"""
        if self.sam_image is None:
            return

        plt.clf()
        plt.imshow(self.sam_image)

        # Display mask if available
        if self.masks is not None:
            mask = self.masks[0]
            overlay = np.zeros_like(self.sam_image, dtype=np.float32)
            overlay[:, :, :] = [255, 255, 255]  # White overlay

            # Create semi-transparent overlay
            alpha = 0.3
            blended = np.where(mask[:, :, None],
                               self.sam_image * (1 - alpha) + overlay * alpha,
                               self.sam_image)
            plt.imshow(blended.astype(np.uint8))

        # Plot points
        if self.foreground_points:
            plt.scatter(*zip(*self.foreground_points), color="red", marker="x",
                        s=50, label="Foreground (Left-Click)")
        if self.background_points:
            plt.scatter(*zip(*self.background_points), color="blue", marker="o",
                        s=50, label="Background (Right-Click)")

        plt.legend()
        plt.axis("off")
        plt.draw()

    def _on_close(self, event):
        """Handle figure closing and save results"""
        if self.masks is None or self.file_destination is None:
            return

        try:
            mask = self.masks[0]

            # Save mask
            mask_path = f"{self.file_destination}_mask.tiff"
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

            # Save masked image
            segmented_image = self.sam_image.copy()
            segmented_image[~mask] = 0
            segmented_path = f"{self.file_destination}_masked.tiff"
            cv2.imwrite(segmented_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

            print(f"Saved mask to: {mask_path}")
            print(f"Saved masked image to: {segmented_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def generate_mask(self, image_path, save_directory):
        """Main method to generate mask for an image"""
        # Reset state
        self.reset_points()
        self.file_destination = save_directory

        # Load image
        if not self.load_image(image_path):
            return False

        # Create figure and connect events
        self.current_figure = plt.figure()
        self.current_figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.current_figure.canvas.mpl_connect("close_event", self._on_close)

        # Show initial image
        plt.imshow(self.sam_image)
        plt.axis("off")
        plt.show()

        return True