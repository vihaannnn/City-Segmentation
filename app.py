import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page title
st.title("Image Segmentation with U-Net")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./unet_model.h5')

model = load_model()

def preprocess_image(image):
    image_list = []
    image_list.append(image)
    image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
    image_filenames = tf.constant(image_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames))

    def process_path(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img

    def preprocess(image, mask):
        input_image = tf.image.resize(image, (96, 128), method='nearest')
        input_image = input_image / 255.
        return input_image

    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)

    return processed_image_ds[0]
    

def perform_segmentation(model, processed_image):
    # Make prediction
    prediction = model.predict(processed_image)
    
    # Convert prediction to binary mask (assuming binary segmentation)
    prediction = (prediction > 0.5).astype(np.uint8)
    prediction = np.squeeze(prediction) * 255
    return prediction

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Sidebar options
st.sidebar.header("Segmentation Parameters")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

if uploaded_file is not None:
    # Create two columns for original and segmented images
    col1, col2 = st.columns(2)
    
    # Display original image
    image = Image.open(uploaded_file)
    with col1:
        st.header("Original Image")
        st.image(image)
    
    # Add segmentation button
    if st.button("Perform Segmentation"):
        with st.spinner("Processing image..."):
            try:
                # Preprocess the image
                processed_img = preprocess_image(image)
                
                # Perform segmentation
                segmentation_mask = perform_segmentation(model, processed_img)
                
                # Display segmentation result
                with col2:
                    st.header("Segmentation Mask")
                    st.image(segmentation_mask, clamp=True)
                
                # Add download button
                mask_image = Image.fromarray(segmentation_mask)
                mask_bytes = mask_image.tobytes()
                st.download_button(
                    label="Download Segmentation Mask",
                    data=mask_bytes,
                    file_name="segmentation_mask.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error during segmentation: {str(e)}")



