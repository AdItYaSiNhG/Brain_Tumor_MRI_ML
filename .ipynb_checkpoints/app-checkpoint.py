import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained Transfer Learning model
# Make sure the path is correct
model_path = 'transfer_learning_mobilenetv2_model.h5' # Or .keras
model = load_model(model_path)

# Define your class names in the correct order (corresponding to your model's output)
# You can get this from test_generator.class_indices
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("Brain Tumor MRI Image Classification")
st.write("Upload an MRI image to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded MRI Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224)) # Match your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescale to 0-1, just like your ImageDataGenerator

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100

    st.success(f"Prediction: {predicted_class_name}")
    st.write(f"Confidence: {confidence:.2f}%")
