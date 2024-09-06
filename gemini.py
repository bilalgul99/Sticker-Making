import cv2
import numpy as np
def detect_and_draw_boundaries(image):
    """Detects edges and draws boundaries on the image, excluding external background.

    Args:
        image: The input image.

    Returns:
        The image with drawn boundaries.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)  # Adjust thresholds   


    # Find contours with RETR_TREE mode to get both external and internal contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and bounding rectangle
    filtered_contours = []
    height, width = image.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        # Check if the contour is large enough
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            # Check if the bounding rectangle is not too close to the edges
            if x > 10 and y > 10 and x + w < width - 10 and y + h < height - 10:
                filtered_contours.append(contour)

    # Draw filtered contours on the original image
    for contour in filtered_contours:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Green color, thickness 2

    return image

# Load the image
image = cv2.imread("image.png")

result = detect_and_draw_boundaries(image)

# Display the result
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()