import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

# Function to load images from various formats
def load_image(input_file):
    if input_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        return cv2.imread(input_file)
    elif input_file.lower().endswith('.pdf'):
        images = convert_from_path(input_file)
        return np.array(images[0])
    else:
        raise ValueError("Unsupported file format")

# Improved object detection with white background removal, including internal regions
def detect_object(image):
    # Step 1: Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Step 4: Dilate the edges to connect nearby edges
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Step 5: Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assumed to be the main object)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask from the main contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [main_contour], 0, 255, -1)
        
        # Use the mask to isolate the object
        object_isolated = cv2.bitwise_and(image, image, mask=mask)
        
        return main_contour, object_isolated
    else:
        return None, image

# Function to draw boundary based on user input
def draw_boundary(image, contour, margin_mm, inside):
    # Convert mm to pixels (assuming 300 DPI, adjust if necessary)
    dpi = 300
    pixels_per_mm = dpi / 25.4
    margin_pixels = int(margin_mm * pixels_per_mm)

    if inside:
        margin_pixels = -margin_pixels

    # Apply offset for boundary expansion/shrinkage
    new_contour = contour + margin_pixels

    # Create a copy of the image to draw the boundary
    output_image = image.copy()
    cv2.drawContours(output_image, [new_contour], -1, (0, 255, 0), 2)

    return output_image

def main():
    # Load image from input (you can replace this with your image path)
    input_file = 'image.png'  # Replace with actual file
    image = load_image(input_file)

    # Detect object and find boundaries
    contour, object_isolated = detect_object(image)
    if contour is None:
        print("No object detected in the image.")
        return
    
    # Display the isolated object
    cv2.imshow('Isolated Object', object_isolated)
    cv2.waitKey(0)

    # Ask user if they want the boundary inside or outside the object
    user_input = input("Would you like the boundary inside or outside the object? (inside/outside): ").strip().lower()
    margin_mm = float(input("By how many millimeters should the boundary be adjusted? "))
    
    inside = True if user_input == 'inside' else False

    # Draw boundary on the image
    result_image = draw_boundary(object_isolated, contour, margin_mm, inside)

    # Show the final image
    cv2.imshow('Final Image with Boundary', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the image
    output_path = 'output_image.png'
    cv2.imwrite(output_path, result_image)
    print(f"Image saved as {output_path}")

if __name__ == "__main__":
    main()
