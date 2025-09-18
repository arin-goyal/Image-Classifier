import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- Configuration ---
# Set the title and icon for the browser tab
st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="wide")

# --- Model Loading ---
# Use caching to load the model only once
@st.cache_resource
def load_keras_model():
    """Load the pre-trained Keras model from disk."""
    try:
        model = load_model('cifar10_classifier.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- UI and Logic ---
def main():
    # --- Header ---
    st.title("🖼️ CNN Image Classifier")
    st.markdown("Upload an image, and this app will classify it as an **animal** or a **traffic**-related object using a trained Convolutional Neural Network.")

    # --- Model and Class Names ---
    model = load_keras_model()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    if model is None:
        st.warning("The model is not loaded. Please ensure 'cifar10_classifier.h5' is in the same directory.")
        return

    # --- Image Upload ---
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns(2)

        with col1:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.info(f"Original image size: {image.size[0]}x{image.size[1]} pixels")

        with col2:
            st.subheader("Classification Result")
            # Add a button to trigger classification
            if st.button("Classify Image"):
                with st.spinner('Classifying...'):
                    # --- Preprocessing and Prediction ---
                    # Resize image to 32x32 (as required by the model)
                    resized_image = image.resize((32, 32))
                    
                    # Convert image to numpy array and normalize
                    img_array = np.array(resized_image) / 255.0
                    
                    # Add a batch dimension
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Make prediction
                    predictions = model.predict(img_array)
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class_name = class_names[predicted_class_index]
                    confidence = np.max(predictions[0]) * 100

                    # Display the result
                    st.success(f"**Predicted Class:** {predicted_class_name}")
                    st.info(f"**Confidence:** {confidence:.2f}%")

                    # Optional: Display prediction probabilities as a bar chart
                    st.subheader("Prediction Probabilities")
                    # Create a dictionary for easier plotting
                    prob_dict = {class_names[i]: predictions[0][i] for i in range(len(class_names))}
                    st.bar_chart(prob_dict)

if __name__ == '__main__':
    main()