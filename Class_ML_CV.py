from PIL import Image
import os
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# Citations:

# 1: February 5th 2025 10:29am ChatGPT4o 
# 2: February 5th 2025 11:27am ChatGPT40. The testing pipeline was generated usining the following chatGPT4o link:
# https://chatgpt.com/share/67aa6a36-178c-8003-9fa1-963f462fb331

def load_lists():
    """load the images into a list, images and masks.
    returns: a list of images and a list of masks"""
    image_path = 'path/to/images/'
    mask_path = 'path/to/masks/'

    # List and sort the filenames
    image_list = sorted(os.listdir(image_path), key=lambda x: int(x.split('.')[0]))
    mask_list = sorted(
        [f for f in os.listdir(mask_path) if f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0])
    )

    # Add the full paths
    image_list = [os.path.join(image_path, i) for i in image_list]
    mask_list = [os.path.join(mask_path, i) for i in mask_list]

    return image_list, mask_list

def load_images():
    """Load the images from the designated folder
    """
    # Define the paths

    # Select an index (N)
    N = 72  # Example index

    # Read the image and corresponding mask
    image_list, mask_list = load_lists()
    img = Image.open(image_list[N])
    mask = Image.open(mask_list[N])

    mask_array = np.array(mask)
    mask_gray_np = np.mean(mask_array[:, :, :3], axis=2)
    mask_gray_np = (mask_gray_np - mask_gray_np.min()) / (mask_gray_np.max() - mask_gray_np.min()) * 255
    plt.imshow(mask_gray_np, cmap='gray')
    plt.title("Normalized Mask")
    plt.show()

    # Display the images inline
    display.display(img)


def segment_KMeans(x):
  """Accepts an image index from the dataset and returns the segmented version
  """
  
  image_list, mask_list = load_lists()
  #Read the image
  image = cv2.imread(image_list[x])

  #Reshape & prep
  pixels = image.reshape((-1, 3))
  pixels = np.float32(pixels)

  #KMeans
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
  k = 6
  _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS) #1 start

  random_colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8) #1 end

  #Map to labs
  segmented_image = random_colors[labels.flatten()] 

  #Show
  segmented_image = segmented_image.reshape(image.shape)

  #cv2_imshow(segmented_image)

  return segmented_image

def match_segmentation_classes(segmented_image, true_mask): #2
    """
    Matches segmentation labels to ground truth labels, handling cases with different numbers of classes.

    Parameters:
    - segmented_image (np.array): Predicted segmentation (H, W).
    - true_mask (np.array): Ground truth mask (H, W).

    Returns:
    - np.array: Segmented image with aligned class labels.
    """

    # Get unique classes
    pred_classes = np.unique(segmented_image)
    true_classes = np.unique(true_mask)

    num_pred = len(pred_classes)
    num_true = len(true_classes)

    # Compute histogram intersection cost matrix
    cost_matrix = np.zeros((num_pred, num_true))

    for i, pred_class in enumerate(pred_classes):
        for j, true_class in enumerate(true_classes):
            pred_mask = (segmented_image == pred_class).astype(np.uint8)
            true_mask_class = (true_mask == true_class).astype(np.uint8)

            intersection = np.logical_and(pred_mask, true_mask_class).sum()
            union = np.logical_or(pred_mask, true_mask_class).sum()

            # Use IoU (or another metric) as a cost function
            iou = intersection / union if union > 0 else 0
            cost_matrix[i, j] = -iou  # Negative because we minimize cost

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create a mapping
    class_mapping = {pred_classes[row]: true_classes[col] for row, col in zip(row_ind, col_ind)}

    # Assign new class labels
    matched_segmented = np.zeros_like(segmented_image)
    for pred_class, true_class in class_mapping.items():
        matched_segmented[segmented_image == pred_class] = true_class

    return matched_segmented


def evaluate_segmentation(segmented_image, true_mask, num_classes): #2
    """
    Evaluate segmentation accuracy using Pixel Accuracy, IoU, and Dice Score.

    Parameters:
    segmented_image (np.array): The predicted segmentation image (H, W).
    true_mask (np.array): The ground truth mask (H, W).
    num_classes (int): Number of segmentation classes.

    Returns:
    dict: Accuracy metrics for each class and overall accuracy.
    """
    iou_scores = []
    dice_scores = []
    class_accuracies = []

    for class_idx in range(num_classes):
        pred_mask = (segmented_image == class_idx).astype(np.uint8)
        gt_mask = (true_mask == class_idx).astype(np.uint8)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        iou = intersection / union if union > 0 else 0
        dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0
        class_accuracy = intersection / gt_mask.sum() if gt_mask.sum() > 0 else 0

        iou_scores.append(iou)
        dice_scores.append(dice)
        class_accuracies.append(class_accuracy)

    overall_accuracy = (segmented_image == true_mask).sum() / true_mask.size

    return {
        "IoU": iou_scores,
        "Class Accuracy": class_accuracies,
        "Overall Accuracy": overall_accuracy
    }

def mask_to_label_array(mask_image): #2
    """
    Converts a color-coded segmentation mask (RGB) into a numerical class label array.

    Parameters:
    - mask_image (np.array): RGB segmentation mask (H, W, 3).

    Returns:
    - np.array: Label array (H, W) with unique class IDs.
    - dict: Color-to-label mapping (for debugging or visualization).
    """
    # Reshape image to a 2D array of pixels
    pixels = mask_image.reshape(-1, 3)

    # Get unique colors (each color represents a different class)
    unique_colors, labels = np.unique(pixels, axis=0, return_inverse=True)

    # Create a mapping from color to class index
    color_to_label = {tuple(color): label for label, color in enumerate(unique_colors)}

    # Reshape the label array back to the original mask shape
    label_mask = labels.reshape(mask_image.shape[:2])

    return label_mask, color_to_label


def eval_on_dataset():
    """example of the above code used to evaluate the segmentation technique on the dataset
    """
    OAs = []

    image_list, mask_list = load_lists()

    for i in range(0,len(image_list),50): #select a few from the dataset for times sake.
        segmented_image = segment_KMeans(i)
        seg_im, _ = mask_to_label_array(segmented_image)

        true_mask_image = cv2.imread(mask_list[i])
        true_mask_image = cv2.cvtColor(true_mask_image, cv2.COLOR_BGR2RGB)  # Convert from OpenCV BGR to RGB
        true_mask, color_mapping = mask_to_label_array(true_mask_image)

        seg_im = match_segmentation_classes(seg_im, true_mask)

        # Evaluate segmentation
        metrics = evaluate_segmentation(seg_im, true_mask, num_classes=len(np.unique(true_mask)))
        
        OAs.append(metrics['Overall Accuracy']) # Can be adapted to make use of other metrics produced.

    OA = np.mean(OAs)
    
    print("Mean Overall Accuracy:", OA)
    