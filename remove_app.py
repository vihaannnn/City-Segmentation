import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from huggingface_hub import hf_hub_download

# Set page title
st.title("Interactive Class Removal Segmentation")

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
    # Remove batch dimension
    prediction = np.squeeze(prediction)
    
    # Get class with highest probability for each pixel
    class_mask = np.argmax(prediction, axis=-1)
    
    # Create a binary mask for pixels to keep (1 for keep, 0 for remove)
    keep_mask = np.ones_like(class_mask, dtype=bool)
    for class_idx in classes_to_remove:
        keep_mask = keep_mask & (class_mask != class_idx)
    
    # Resize mask to match original image size
    keep_mask = tf.image.resize(
        tf.expand_dims(tf.expand_dims(keep_mask.astype(float), -1), 0),
        image.shape[:2],
        method='nearest'
    )
    keep_mask = np.squeeze(keep_mask)
    
    # Apply mask to original image
    result = image.copy()
    # Create a boolean mask array matching the image dimensions
    mask_3d = np.stack([keep_mask] * 3, axis=-1)
    # Use boolean indexing to set pixels to white
    result[mask_3d == 0] = 255
    
    return result

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
    st.text(prediction.shape)
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
                result_image = remove_classes(image_array, prediction, classes_to_remove)
                
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

    print(prediction.shape)