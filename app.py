import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# App setup
st.set_page_config(page_title="Potato Disease Classifier", layout="centered")
st.title("🥔 Potato Leaf Disease Detection from Camera or Upload")

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "potato_v1.keras")
    return tf.keras.models.load_model(model_path)

model = load_model()

# Define your class names (adjust if different)
class_names = ['Early Blight', 'Healthy', 'Late Blight']

# Preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

# Camera input or upload
st.subheader("📷 Take a photo OR 📁 upload one")

image_data = st.camera_input("Take a photo using your camera")
uploaded_file = st.file_uploader("Or upload a potato leaf image", type=["jpg", "jpeg", "png"])

# Choose whichever is provided
final_image = None

if image_data is not None:
    # Convert camera image to PIL
    final_image = Image.open(image_data)
elif uploaded_file is not None:
    final_image = Image.open(uploaded_file)

# If there's an image, process it
if final_image:
    st.image(final_image, caption="🖼 Image to be classified", use_column_width=True)
    st.write("🔍 Classifying...")

    processed = preprocess_image(final_image)
    prediction = model.predict(processed)

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    st.success(f"✅ Predicted Class: {predicted_class}")
    st.info(f"🔢 Confidence: {confidence}%")
