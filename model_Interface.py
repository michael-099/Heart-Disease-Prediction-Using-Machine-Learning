import pandas as pd
import joblib
import streamlit as st 

model = joblib.load('Heart_Disease_logistic_regression_model.pkl')

def predict_heart_disease():
    st.title("Heart Disease Prediction Input Form")

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

    input_data = {
        "Age": age,
        "Sex": sex,
        "Chest Pain Type": chest_pain_type,
        "Resting BP": resting_bp,
        "Cholesterol": cholesterol,
        "Fasting BS": fasting_bs,
        "Resting ECG": resting_ecg,
        "Max HR": max_hr,
        "Exercise Angina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST Slope": st_slope
    }

    st.write("User Inputs:")
    st.json(input_data)

    input_data = pd.DataFrame([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, 
                                resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]], 
                              columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                                       'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                                       'Oldpeak', 'ST_Slope'])

    input_data_array = input_data.values
    prediction = model.predict(input_data_array)

    if st.button("Predict Heart Disease"):
        if prediction[0] == 1:  
            st.write("🚨 **The model predicts: Heart Disease** 🚨")
            st.write("Based on the input data, the model has identified a high likelihood of heart disease. Please consult a healthcare professional for further evaluation and testing.")
            st.write("The model suggests immediate action or lifestyle changes to mitigate potential risks.")
        else:
            st.write("✅ **The model predicts: No Heart Disease** ✅")
            st.write("Based on the input data, the model indicates a low likelihood of heart disease. However, maintaining a healthy lifestyle and regular check-ups are recommended to stay healthy.")

if __name__ == '__main__':
    predict_heart_disease()
