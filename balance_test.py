import cv2

filename = './data/balanced8/target_southwest.tiff'
# Load the image
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Apply CLAHE
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#image_clahe = clahe.apply(image)

# Save the output image
cv2.imwrite(filename, image)