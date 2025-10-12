import cv2
import numpy as np
import os

def scan_image_for_yellow_hsv_range(image_path, lower_hue=20, upper_hue=40, min_saturation=80, min_value=60):
    """
    Scans a given image to identify pixels within a specified HSV range for yellow color.

    Args:
        image_path (str): The path to the image file.
        lower_hue (int): The lower limit for the hue value (0-179).
                         Typical yellow hue is around 20-40.
        upper_hue (int): The upper limit for the hue value (0-179).
        min_saturation (int): The minimum saturation value (0-255).
                               Pixels must have saturation >= this value.
        min_value (int): The minimum value/brightness value (0-255).
                         Pixels must have value >= this value.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from '{image_path}'. Check file format or corruption.")
        return

    print(f"Scanning image: '{image_path}' for pixels within the following HSV range:")
    print(f"  Hue (H): {lower_hue} to {upper_hue}")
    print(f"  Saturation (S): {min_saturation} to 255")
    print(f"  Value (V): {min_value} to 255")


    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the HSV range
    lower_bound = np.array([lower_hue, min_saturation, min_value])
    upper_bound = np.array([upper_hue, 255, 255])

    # Create a binary mask where pixels falling within the HSV range are white (255)
    # and others are black (0).
    hsv_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Count the number of white pixels in the mask
    yellow_pixel_count = np.sum(hsv_mask) // 255

    if yellow_pixel_count > 0:
        print(f"\nMessage: Found {yellow_pixel_count} pixels matching the specified HSV criteria in '{image_path}'.")
    else:
        print(f"\nMessage: No pixels found matching the specified HSV criteria in '{image_path}'.")

    # --- Visual Feedback ---
    # Display the original image
    cv2.imshow("Original Image", image)

    # Display the HSV mask (white where pixels match the criteria, black otherwise)
    cv2.imshow(f"Yellow HSV Filtered Mask (H:{lower_hue}-{upper_hue}, S>={min_saturation}, V>={min_value})", hsv_mask)

    print("\nPress any key to close the image windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- How to use YOUR image with Saturation and Value restrictions ---

    # 1. Place an image file containing your **yellow-colored chess pieces**
    #    in the same directory as this Python script, OR provide the full path.

    # 2. Change the value of 'my_image_file_path' below to your image's filename
    #    (e.g., 'yellow_piece_photo.jpg').
    #    If your image is in a different folder, provide the full path:
    #    e.g., 'C:/Users/YourUsername/Pictures/yellow_piece_photo.jpg' (Windows)
    #    e.g., '/home/yourusername/Desktop/my_yellow_pieces.png' (Linux/macOS)

    my_image_file_path = "6.jpg" # <--- **CHANGE THIS LINE** (Use an image with your yellow pieces)

    # 3. Adjust the HSV parameters for yellow:
    #    - lower_hue & upper_hue: These define the range for yellow. Typical yellow hues are
    #      often between 20-40. Start with these and adjust.
    #    - min_saturation: Raise this value (e.g., 50, 80, 100) to filter out grayish/dull yellows.
    #                      Decrease it if your yellow piece looks faded.
    #    - min_value: Raise this value (e.g., 50, 80, 100) to filter out very dark/shadowed areas.
    #                 Decrease it if your yellow piece looks dark.

    my_lower_yellow_hue = 15
    my_upper_yellow_hue = 70
    my_min_yellow_saturation = 70  # Example: Require saturation to be at least 80 (out of 255)
    my_min_yellow_value = 50       # Example: Require brightness to be at least 60 (out of 255)

    # Call the function with your image path and the adjusted HSV parameters
    scan_image_for_yellow_hsv_range(
        image_path=my_image_file_path,
        lower_hue=my_lower_yellow_hue,
        upper_hue=my_upper_yellow_hue,
        min_saturation=my_min_yellow_saturation,
        min_value=my_min_yellow_value
    )

    # You can uncomment the dummy image block below for quick testing if you don't have a photo ready.
    # dummy_image_path = "dummy_yellow_hsv_test.png"
    # dummy_img = np.zeros((200, 300, 3), dtype=np.uint8)
    # # Bright Yellow (BGR - Note: BGR for yellow is typically [0, 255, 255])
    # dummy_img[:, :100] = [0, 255, 255]
    # # Dark Yellow (BGR)
    # dummy_img[:, 100:200] = [0, 100, 100]
    # # Desaturated Yellow (BGR)
    # dummy_img[:, 200:] = [100, 200, 200]
    # cv2.imwrite(dummy_image_path, dummy_img)
    # print(f"\nCreated a dummy image at: {dummy_image_path} for demonstration.")
    # scan_image_for_yellow_hsv_range(
    #     image_path=dummy_image_path,
    #     lower_hue=20,
    #     upper_hue=40,
    #     min_saturation=80,
    #     min_value=60
    # )
    # if os.path.exists(dummy_image_path):
    #     os.remove(dummy_image_path)
    #     print(f"Cleaned up dummy image: {dummy_image_path}")
