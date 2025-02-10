import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from huggingface_hub import hf_hub_download
import torch
from diffusers import StableDiffusionInpaintPipeline
import os

# Set page title
st.title("Interactive Class Removal Segmentation")
# Set torch path explicitly
torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]

# Load the model
@st.cache_resource
def load_model(repo_id = "vihaannnn/City_Segmentation_UNet", model_filename = "unet_model.h5"):
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        revision="main"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# def get_pil_image_normalized(image_tensor):
#     """
#     Convert an image tensor to a PIL image by normalizing its pixel values to the full [0,255] range.
#     """
#     image_np = image_tensor.numpy()
#     # Perform min-max normalization to scale pixel values to [0, 1]
#     normalized = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
#     # Scale normalized image to [0,255] and convert to uint8
#     image_8bit = (normalized * 255).astype(np.uint8)
#     return Image.fromarray(image_8bit)

def preprocess_image(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Resize
    img_tensor = tf.image.resize(img_tensor, (192, 256), method='nearest')
    
    # Normalize
    img_tensor = img_tensor / 255.0
    
    # Add batch dimension
    img_tensor = tf.expand_dims(img_tensor, 0)
    
    
    return img_tensor, img_array.shape[:2]

def get_present_classes(prediction):
    # Remove batch dimension
    prediction = np.squeeze(prediction)
    
    # Get class with highest probability for each pixel
    class_mask = np.argmax(prediction, axis=-1)
    
    # Get unique classes
    present_classes = np.unique(class_mask)
    
    return present_classes

def remove_classes(image, prediction, classes_to_remove):
    # First normalize and resize the input image to 192*256
    normalized_image = image.astype(float) / 255.0  # Normalize to [0, 1]
    resized_image = tf.image.resize(
        tf.expand_dims(normalized_image, 0),
        [192, 256],
        method='bilinear'
    )
    resized_image = np.squeeze(resized_image)
    
    # Remove batch dimension from prediction
    prediction = np.squeeze(prediction)
    
    # Get class with highest probability for each pixel
    class_mask = np.argmax(prediction, axis=-1)
    
    # Create a binary mask for pixels to keep (1 for keep, 0 for remove)
    keep_mask = np.ones_like(class_mask, dtype=bool)
    for class_idx in classes_to_remove:
        keep_mask = keep_mask & (class_mask != class_idx)
    
    # Convert mask to float and add necessary dimensions
    keep_mask = tf.expand_dims(tf.expand_dims(keep_mask.astype(float), -1), 0)
    keep_mask = np.squeeze(keep_mask)
    
    # Apply mask to resized image
    result = resized_image.copy()
    # Create a boolean mask array matching the image dimensions
    mask_3d = np.stack([keep_mask] * 3, axis=-1)
    # Use boolean indexing to set pixels to white
    result[mask_3d == 0] = 1.0  # Set to 1.0 (white) in normalized space
    
    # Denormalize back to [0, 255] range
    result = (result * 255.0).astype(np.uint8)
    
    return result, keep_mask

def mask_preprocesing(predicted_mask):
    # take the channel index with the highest value
    reduced_channeled_predicted_mask = tf.argmax(predicted_mask, axis=-1)
    

    # have the channel dimension back, instead of being 23 (num of classes) but be 1
    # Now, segmentation_mask is a tensor where each pixel value indicates the predicted class
    # (an integer in the range 0–22).
    final_pred_mask = reduced_channeled_predicted_mask[..., tf.newaxis]

    # Convert the TensorFlow tensor to a NumPy array and remove the channel dimension
    seg_mask_np = final_pred_mask.numpy().squeeze()

    return seg_mask_np
    

def get_binary_mask(segmentation_mask, target_class):
    """
    Convert a multi-class segmentation mask to a binary mask for a specific target class.
    Pixels with the target_class are set to 255 (white) and all others to 0 (black).
    """
    # Convert the TensorFlow tensor to a NumPy array and remove the channel dimension
    seg_mask_np = segmentation_mask.squeeze()  # Shape becomes (H, W)
    # Create binary mask: True where the pixel equals target_class, then convert to 255/0
    binary_mask = (seg_mask_np == target_class).astype(np.uint8) * 255
    # Convert the NumPy array to a PIL image (the inpainting pipeline expects a PIL image)
    return Image.fromarray(binary_mask)

def combine_binary_masks(binary_masks):
    """
    Combine multiple binary masks into a single binary mask.
    All masks must have the same shape.
    
    Args:
        binary_masks (list): List of PIL Image binary masks where white (255) represents
                           the area of interest and black (0) represents background
    
    Returns:
        PIL.Image: Combined binary mask where any white pixel from any input mask
                  remains white in the final mask
    """
    if not binary_masks:
        raise ValueError("No masks provided to combine")
    
    # Convert first mask to numpy array to get shape
    first_mask = np.array(binary_masks[0])
    combined_mask = np.zeros_like(first_mask)
    
    # Combine all masks using logical OR operation
    for mask in binary_masks:
        mask_array = np.array(mask)
        # If any pixel is white (255) in any mask, make it white in the combined mask
        combined_mask = np.logical_or(combined_mask, mask_array > 0).astype(np.uint8) * 255
    
    # Convert back to PIL Image
    return Image.fromarray(combined_mask), combined_mask

def diffuse_image(predicted_mask, pil_image, target_classes):
    segmentation_mask = mask_preprocesing(predicted_mask)
    binary_masks = []
    for target_class in target_classes:
        binary_mask = get_binary_mask(segmentation_mask, target_class)
        binary_masks.append(binary_mask)
    
    combined_mask, combined_mask_numpy = combine_binary_masks(binary_masks)
    st.image(combined_mask)
    st.text("BINARY MASK")
    str = combined_mask_numpy.shape
    st.text(str)
    # Load the inpainting pipeline. Use fp16 for faster inference if a GPU is available.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    )

    # Move the pipeline to the GPU if available; otherwise, it will run on the CPU.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipe.to(device)

    # Define your text prompt
    prompt = "decorated sky"
    # Run the inpainting process
    st.text("FINALLL IMAGE AND MASK")
    st.image(pil_image)
    
    st.text("shape of image ")
    image_np = np.array(pil_image).shape
    st.text(image_np)


    st.image(combined_mask)
    st.text("shape of mask ")
    mask_np = np.array(combined_mask).shape
    st.text(mask_np)
    result = pipe(prompt=prompt, image=pil_image, mask_image=binary_mask).images[0]
    # Convert the PIL image 'result' to a TensorFlow tensor
    result_tensor = tf.keras.preprocessing.image.img_to_array(result)
    # Resize using TensorFlow (note: the size argument is in (height, width))
    result_tensor_resized = tf.image.resize(result_tensor, [192, 256])
    print(result_tensor_resized.shape)
    # Convert tensorflow tensor back to a PIL image
    diffused_image = tf.keras.preprocessing.image.array_to_img(result_tensor_resized)
    # Now call the function with your four images:
    # pil_image: your original input image (tensor or PIL image)
    # final_pred_mask: your predicted segmentation mask (tensor or NumPy array, shape (H, W, C))
    # binary_mask: your binary mask for the selected object (already a PIL image)
    # result: your diffused (inpainted) image (tensor, NumPy array, or PIL image)
    st.text("Diffused Image")
    st.image(diffused_image)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Create two columns for original and processed images
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        st.image(image)
    
    # Process image to get present classes
    processed_img, original_size = preprocess_image(image)
    prediction = model.predict(processed_img)
    present_classes = get_present_classes(prediction)
    
    # Show checkboxes only for present classes
    st.header("Select classes to remove:")
    cols = st.columns(4)
    classes_to_remove = []
    
    # Create checkboxes only for present classes
    for i, class_idx in enumerate(present_classes):
        with cols[i % 4]:
            if st.checkbox(f"Class {class_idx}", key=f"class_{class_idx}"):
                classes_to_remove.append(class_idx)
    
    # Show class distribution
    class_mask = np.argmax(np.squeeze(prediction), axis=-1)
    unique, counts = np.unique(class_mask, return_counts=True)
    st.write("Class Distribution:")
    for cls, count in zip(unique, counts):
        total_pixels = class_mask.size
        percentage = (count / total_pixels) * 100
        st.write(f"Class {cls}: {count} pixels ({percentage:.1f}%)")
    
    # Add process button
    if st.button("Process Image"):
        with st.spinner("Processing image..."):
            try:
                # Remove selected classes
                result_image, keep_mask = remove_classes(image_array, prediction, classes_to_remove)
                st.text(keep_mask.shape)
                
                # Display result
                with col2:
                    st.header("Processed Image")
                    st.image(result_image)
                
                # Add download button for processed image
                result_pil = Image.fromarray(result_image.astype('uint8'))
                buf = io.BytesIO()
                result_pil.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Processed Image",
                    data=byte_im,
                    file_name="processed_image.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")


    # Resize to specific dimensions
    # st.image(image)
    # pil_image = image.resize((192, 256))
    # st.image(pil_image)
    # reshaped_image_pil= np.array(pil_image, dtype=np.float32).transpose(1, 0, 2)
    # pil_image = Image.fromarray((reshaped_image_pil * 255).astype('uint8'))
    # st.image(pil_image)


    # st.text(np.array(pil_image, dtype=np.float32).shape)


    # Create lists of file paths instead of single strings
    import os

    # Get all files in the directories
    image_filenames = [os.path.join("./CameraRGB", f) for f in os.listdir("./CameraRGB") if f.endswith(('.png', '.jpg', '.jpeg'))]
    masks_filenames = [os.path.join("./Masks", f) for f in os.listdir("./Masks") if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort to ensure matching pairs
    image_filenames.sort()
    masks_filenames.sort()

    # Convert to tensor
    image_filenames = tf.constant(image_filenames)
    masks_filenames = tf.constant(masks_filenames)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

    for image, mask in dataset.take(2):
        print(image)
        print(mask)

    multiplier = 2
    def process_path(image_path, mask_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
        return img, mask

    def preprocess(image, mask):
        input_image = tf.image.resize(image, (96*multiplier, 128*multiplier), method='nearest')
        input_mask = tf.image.resize(mask, (96*multiplier, 128*multiplier), method='nearest')

        input_image = input_image / 255.

        return input_image, input_mask

    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)


    for image, mask in processed_image_ds.take(1):
        sample_image, sample_mask = image, mask
        print(mask.shape)
    

    def get_pil_image_normalized(image_tensor):
        """
        Convert an image tensor to a PIL image by normalizing its pixel values to the full [0,255] range.
        """
        image_np = image_tensor.numpy()
        # Perform min-max normalization to scale pixel values to [0, 1]
        normalized = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
        # Scale normalized image to [0,255] and convert to uint8
        image_8bit = (normalized * 255).astype(np.uint8)
        print(image_8bit.shape)
        return Image.fromarray(image_8bit)

    # Use the normalized function to get a PIL image
    pil_image = get_pil_image_normalized(sample_image)
    pil_image  # This should display the image with a stretched contrast.
    diffuse_image(predicted_mask=prediction, pil_image=pil_image , target_classes=[6,7])

    print(prediction.shape)