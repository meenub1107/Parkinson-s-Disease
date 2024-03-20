import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the models
spiral_model_path =r"C:\Users\meenu\OneDrive\Desktop\Parkinsons\model.h5"  
wave_model_path = r"C:\Users\meenu\OneDrive\Desktop\Parkinsons\classifier.h5"  
spiral_model = load_model(spiral_model_path)
wave_model = load_model(wave_model_path)

# Define the function to load and predict the image
def predict_image(img, model):
    img = img.convert("RGB")  # Convert image to RGB mode
    img = img.resize((128, 128))  # Resize the image to match the model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    
    if predicted_class_index == 0:
        return "Parkinson"
    else:
        return "Healthy"
    
# Streamlit app
def main():
    st.title("Parkinson vs. Healthy Prediction")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # Make prediction for spiral model when button is clicked
        if st.button("Predict Spiral Model"):
            prediction_spiral = predict_image(image, spiral_model)
            st.success(f"The predicted class using Spiral Model is: {prediction_spiral}")
        
        # Make prediction for wave model when button is clicked
        if st.button("Predict Wave Model"):
            prediction_wave = predict_image(image, wave_model)
            st.success(f"The predicted class using Wave Model is: {prediction_wave}")

if __name__ == "__main__":
    main()