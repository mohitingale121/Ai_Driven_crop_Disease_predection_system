import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the model once at the start
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()  # Load the model at the beginning

# Load Disease and Supplement Information
@st.cache_data
def load_disease_data():
    disease_info = pd.read_csv("disease_info.csv", encoding='ISO-8859-1')  # Change encoding to ISO-8859-1
    supplement_info = pd.read_csv('supplement_info.csv')
    return disease_info, supplement_info

disease_info, supplement_info = load_disease_data()

# Tensorflow Model Prediction
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Retrieve disease and supplement information
def get_disease_info(disease_name):
    # Check if the disease name exists in the dataset
    disease_rows = disease_info[disease_info['disease_name'] == disease_name]  # Update 'Disease' to 'disease_name'
    
    if not disease_rows.empty:
        # If disease is found, return the description and supplements
        disease_description = disease_rows['description'].values[0]
        possible_steps = disease_rows['Possible Steps'].values[0]
        image_url = disease_rows['image_url'].values[0]
    else:
        # If disease is not found, return a default message
        disease_description = "Description not available for this disease."
        possible_steps = "No steps available."
        image_url = "No image available."

    # Fetch supplement details, if any
    supplement_details = supplement_info[supplement_info['Disease'] == disease_name]

    return disease_description, possible_steps, image_url, supplement_details

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown(""" 
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set while preserving the directory structure.
                
                #### Content
                1. Train (70,295 images)
                2. Test (33 images)
                3. Validation (17,572 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    # Show small scale version of the uploaded image
    if test_image and st.button("Show Image"):
        st.image(test_image, caption="Uploaded Image", width=250)

    # Predict button
    if test_image and st.button("Predict"):
        with st.spinner("Processing the image..."):
            result_index = model_prediction(test_image)

        # Define class names
        class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry___Powdery_mildew',
            'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        # Get the predicted disease name
        disease_name = class_names[result_index]
        st.success(f"Model is predicting it's a {disease_name}")

        # Display the disease name being processed
        st.write(f"Disease Name being processed: {disease_name}")

        # Get disease description, possible steps, and image URL
        disease_description, possible_steps, image_url, supplement_details = get_disease_info(disease_name)

        # Display disease information
        st.subheader("Disease Information")
        st.write(disease_description)
        st.write("**Possible Steps**: " + possible_steps)
        
        if image_url and image_url != 'No image available.':
            st.image(image_url, caption="Disease Image", width=400)
        else:
            st.write("No image available.")

        # Display recommended supplements
        st.subheader("Recommended Supplements")
        if not supplement_details.empty:
            # Adjust column names based on your CSV file
            for _, supplement in supplement_details.iterrows():
                # Ensure column names match what's in the CSV
                supplement_name = supplement.get('supplement_name', 'Name not available')
                supplement_image = supplement.get('supplement image', None)  # Use None if no image is provided
                supplement_link = supplement.get('buy link', '#')

                st.write(f"**Supplement Name:** {supplement_name}")

                # Display supplement image if available
                if supplement_image and supplement_image != 'No image available':
                    st.image(supplement_image, width=100)
                else:
                    st.write("No image available.")

                st.markdown(f"[Buy Link]({supplement_link})")
        else:
            st.write("No recommended supplements available.")
