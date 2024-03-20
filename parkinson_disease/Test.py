import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

def main():
    st.title("Parkinson's Disease Prediction")

    # File uploader for an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        prediction_type = st.selectbox("Select prediction type", ["Wave", "Spiral"])

        if prediction_type == "Wave":
            print('Wave')
            model_path =r"C:\Users\meenu\OneDrive\Desktop\Parkinsons\classifier.h5"  
        elif prediction_type == "Spiral":
            print('Spiral')
            model_path =r"C:\Users\meenu\OneDrive\Desktop\Parkinsons\model.h5"  
        else:
            st.error("Invalid prediction type.")
            return

        model = load_model(model_path)

        # Define the class labels
        class_labels = ["Healthy", "Parkinson's Disease"]

        # Predict and display result when button is clicked
        predict_button = st.button("Predict")

        if predict_button:
            # Preprocess the image
            img_array = np.array(image.resize((128, 128))) / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)

            # Predict using the model
            prediction = model.predict(img_tensor)

            # Get the predicted class index
            predicted_class_index = np.argmax(prediction[0])

            # Display prediction result
            st.success(f"The image is predicted to be {class_labels[predicted_class_index]}.")

if __name__ == "__main__":
    main()