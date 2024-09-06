import cv2
import numpy as np

def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [main_contour], 0, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def color_flood_fill(image, seed_point, tolerance=32):
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    flood_fill_flags = (
        4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    )
    
    cv2.floodFill(image, mask, seed_point, (255, 255, 255), 
                  (tolerance, tolerance, tolerance), (tolerance, tolerance, tolerance),
                  flood_fill_flags)
    
    return mask[1:-1, 1:-1]

def on_mouse(event, x, y, flags, params):
    global image_no_bg, mask, is_drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        color_mask = color_flood_fill(image_no_bg, (x, y))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(color_mask))
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        color_mask = color_flood_fill(image_no_bg, (x, y))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(color_mask))
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

# Load the image
image = cv2.imread("image.png")

# Remove background
image_no_bg = remove_background(image)

# Create initial mask
mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

# Create a window for interactive display
cv2.namedWindow("Interactive Image")
cv2.setMouseCallback("Interactive Image", on_mouse)

is_drawing = False

while True:
    # Apply the current mask to the image
    display_image = cv2.bitwise_and(image_no_bg, image_no_bg, mask=mask)
    
    # Draw a semi-transparent overlay to show selected areas
    overlay = display_image.copy()
    cv2.addWeighted(
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 
        overlay, 0.5, 0, overlay
    )

    cv2.imshow("Interactive Image", overlay)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset selection
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

# Create the final image
final_image = cv2.bitwise_and(image_no_bg, image_no_bg, mask=mask)

cv2.imshow("Final Image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final image
cv2.imwrite("final_image.png", final_image)