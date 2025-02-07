import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Set page title
st.title("Interactive Class Removal Segmentation")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./unet_model.h5')

model = load_model()

def preprocess_image(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Resize
    img_tensor = tf.image.resize(img_tensor, (96, 128), method='nearest')
    
    # Normalize
    img_tensor = img_tensor / 255.0
    
    # Add batch dimension
    img_tensor = tf.expand_dims(img_tensor, 0)
    
    return img_tensor, img_array.shape[:2]

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

# Create columns for checkboxes (4 columns to make it more compact)
if uploaded_file is not None:
    st.header("Select classes to remove:")
    cols = st.columns(4)
    classes_to_remove = []
    
    # Create checkboxes for each class
    for i in range(23):
        with cols[i % 4]:
            if st.checkbox(f"Class {i}", key=f"class_{i}"):
                classes_to_remove.append(i)
    
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
    
    # Add process button
    if st.button("Process Image"):
        with st.spinner("Processing image..."):
            try:
                # Preprocess the image
                processed_img, original_size = preprocess_image(image)
                
                # Get model prediction
                prediction = model.predict(processed_img)
                
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