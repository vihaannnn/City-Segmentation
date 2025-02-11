import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import IPython.display as display

#citations

# some code was reused from the Classical_ML script

# Convert RGB image to grayscale
def rgb_to_grayscale(image):
    image_array = np.array(image)
    return np.mean(image_array, axis=-1, keepdims=True)

# Group pixels into classes based on threshold
def group_pixels(grayscale_image, threshold=30):
    classes = (grayscale_image / threshold).astype(int)
    return np.clip(classes, 0, 255 // threshold)

# Rebuild the image with different colors for each class
def rebuild_image(class_mask, num_classes):
    colors = np.random.randint(0, 256, (num_classes, 3), dtype=np.uint8)  # Random colors for each class
    color_image = colors[class_mask.squeeze()]  # Remove any unnecessary singleton dimensions
    return color_image

def load_path_list():
    image_path = 'images/'
    image_list = sorted(os.listdir(image_path), key=lambda x: int(x.split('.')[0]))
    image_list = [os.path.join(image_path, i) for i in image_list]
    return image_list

def example_use():
    N = 72  # Example index

    # Read the image 
    image_list = load_path_list()
    img = Image.open(image_list[N])

    # Convert to greyscale
    gray_image = rgb_to_grayscale(img)

    # Group pixels into classes
    threshold = 30
    class_mask = group_pixels(gray_image, threshold)

    print("Class mask shape:", class_mask.shape)  # Should match original dimensions (e.g., (H, W, 1))

    # Rebuild and display the image
    num_classes = (255 // threshold) + 1
    reconstructed_image = rebuild_image(class_mask, num_classes)

    plt.imshow(reconstructed_image)
    plt.title("Reconstructed Image with Colored Classes")
    plt.axis('off')
    plt.show()

    # Display original image
    display.display(img)