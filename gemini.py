import cv2
import numpy as np

def detect_and_draw_boundaries(image):
    """Detects edges and draws boundaries on the image.

    Args:
        image: The input image.

    Returns:
        The image with drawn boundaries.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)  # Adjust thresholds as needed

    # Find contours with RETR_TREE mode to get both external and internal contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    for contour in contours:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Green color, thickness 2

    return image

# Load the image
image = cv2.imread("image.png")

# Detect edges and draw boundaries
result = detect_and_draw_boundaries(image)

# Display the result
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()