import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils import variables

# warnings
import warnings
warnings.filterwarnings('ignore')

# get the root directory
root_dir = os.path.dirname(os.path.abspath(__file__))


# # Load the trained model
model = load_model(os.path.join(root_dir, "model", "breast_cancer.h5"))
print(model)


scaler = joblib.load(os.path.join(root_dir, "model", "scaler.pkl")) 
print(scaler)

# Streamlit App
def main():
    # Title of the app
    st.title(variables.app_name)
    
    # Instructions
    st.markdown("This app predicts whether a tumor is malignant or benign based on input features.")
    
    # Create input fields for each feature
    worst_concave_points = st.slider('Worst Concave Points', 0.0, 0.2, 0.1)
    worst_perimeter = st.slider('Worst Perimeter', 50.0, 250.0, 100.0)
    mean_concave_points = st.slider('Mean Concave Points', 0.0, 0.2, 0.1)
    worst_radius = st.slider('Worst Radius', 10.0, 30.0, 20.0)
    mean_perimeter = st.slider('Mean Perimeter', 50.0, 200.0, 100.0)
    worst_area = st.slider('Worst Area', 200.0, 2000.0, 1000.0)
    mean_radius = st.slider('Mean Radius', 10.0, 30.0, 20.0)
    mean_area = st.slider('Mean Area', 100.0, 2000.0, 1000.0)
    mean_concavity = st.slider('Mean Concavity', 0.0, 0.3, 0.1)
    worst_concavity = st.slider('Worst Concavity', 0.0, 0.3, 0.1)

    # Create a dataframe from the input values
    user_input = np.array([[worst_concave_points, worst_perimeter, mean_concave_points, worst_radius,
                            mean_perimeter, worst_area, mean_radius, mean_area, mean_concavity, worst_concavity]])
    
    # Scale the input data
    user_input_scaled = scaler.transform(user_input)

    # Prediction
    prediction = model.predict(user_input_scaled)
    
    # Output the prediction result
    if prediction < 0.5:
        st.markdown("The tumor is **Benign**.")
    else:
        st.markdown("The tumor is **Malignant**.")
        
    # Optional: Displaying the prediction probability
    st.write(f"Prediction Probability: {prediction[0][0]:.2f}")

# Run the app
if __name__ == '__main__':
    main()