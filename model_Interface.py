import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

try:
    model = joblib.load('Heart_Disease_logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler. Details: {str(e)}")
    st.stop()

def predict_heart_disease():
    st.title("Heart Disease Prediction Input Form")

    # Input fields
    age = st.number_input("Enter age (in years): ", min_value=1, max_value=120, step=1)
    sex = st.radio("Enter sex", options=[1, 2], index=0, format_func=lambda x: "Male" if x == 1 else "Female")
    chest_pain_type = st.selectbox(
        "Enter chest pain type", 
        options=[1, 2, 3, 4], 
        format_func=lambda x: {1: "ATA", 2: "NAP", 3: "ASY", 4: "TA"}[x]
    )
    resting_bp = st.number_input("Enter resting blood pressure (in mmHg): ", min_value=50, max_value=200, step=1)
    cholesterol = st.number_input("Enter cholesterol level (in mg/dl): ", min_value=100, max_value=600, step=1)
    fasting_bs = st.radio("Enter fasting blood sugar", options=[0, 1], index=0, format_func=lambda x: "False" if x == 0 else "True")
    resting_ecg = st.selectbox(
        "Enter resting electrocardiographic results", 
        options=[1, 2, 3], 
        format_func=lambda x: {1: "Normal", 2: "ST", 3: "LVH"}[x]
    )
    max_hr = st.number_input("Enter maximum heart rate achieved (in beats per minute): ", min_value=50, max_value=220, step=1)
    exercise_angina = st.radio("Enter exercise-induced angina", options=[1, 2], index=0, format_func=lambda x: "No" if x == 1 else "Yes")
    oldpeak = st.number_input("Enter depression induced by exercise relative to rest (in depression units): ", min_value=0.0, max_value=6.0, step=0.1)
    st_slope = st.selectbox(
        "Enter slope of the peak exercise ST segment", 
        options=[1, 2, 3], 
        format_func=lambda x: {1: "Up", 2: "Flat", 3: "Down"}[x]
    )

    # Create DataFrame for input data
    input_data = pd.DataFrame(
        [[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, 
          max_hr, exercise_angina, oldpeak, st_slope]], 
        columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    )

    st.write("User Inputs:")
    st.json(input_data.to_dict(orient="records")[0])
    def show_disclaimer():
        st.markdown(
            """
            ---
            ### ⚠ Important Disclaimer  
            This application uses a machine learning model to provide predictions based on the data you input.  

            - **Not 100% Reliable:** The model is not guaranteed to be accurate and may produce false positives or negatives.  
            - **Not for Medical Use:** This tool is for educational and informational purposes only and is **not a substitute for professional medical advice, diagnosis, or treatment**.  
            - **Seek Professional Advice:** Always consult a qualified healthcare provider for medical concerns or decisions.  

            **Use this tool responsibly and at your own discretion.**
            ---
            """
    )

    # Prediction logic
    if st.button("Predict Heart Disease"):
        try:
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)
            
            if prediction[0] == 1:
                st.markdown(
        """
        🚨 **Prediction Result: Heart Disease Detected** 🚨  
        Based on the input data provided, the model predicts that the individual is likely to have heart disease.  
        Please consult with a healthcare professional for further diagnosis and advice.
        """
                 )
            else:
                st.markdown(
        """
        ✅ **Prediction Result: No Heart Disease Detected** ✅  
        Based on the input data provided, the model predicts that the individual is unlikely to have heart disease.  
        Maintain a healthy lifestyle and consult with a healthcare professional for regular check-ups.
        """
                
                )

        except Exception as e:
            st.error(f"Error during prediction. Details: {str(e)}")
        show_disclaimer()

if __name__ == '__main__':
    predict_heart_disease()
