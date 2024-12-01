import pandas as pd
import joblib

# Load the trained model
model = joblib.load('Heart_Disease_logistic_regression_model.pkl')

def predict_heart_disease():
   
    age = int(input("Enter age (in years): "))  
    sex = int(input("Enter sex (1 = Male, 2 = Female): "))  
    chest_pain_type = int(input("Enter chest pain type (1 = ATA, 2 = NAP, 3 = ASY, 4 = TA): "))
    resting_bp = int(input("Enter resting blood pressure (in mmHg): "))  
    cholesterol = int(input("Enter cholesterol level (in mg/dl): "))  
    fasting_bs = int(input("Enter fasting blood sugar (1 = True, 0 = False): ")) 
    resting_ecg = int(input("Enter resting electrocardiographic results (1 = Normal, 2 = ST, 3 = LVH): "))
    max_hr = int(input("Enter maximum heart rate achieved (in beats per minute): ")) 
    exercise_angina = int(input("Enter exercise-induced angina (1 = No, 2 = Yes): "))  
    oldpeak = float(input("Enter depression induced by exercise relative to rest (in depression units): "))  
    st_slope = int(input("Enter slope of the peak exercise ST segment (1 = Up, 2 = Flat, 3 = Down): "))

  
    input_data = pd.DataFrame([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, 
                                resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]], 
                              columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                                       'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                                       'Oldpeak', 'ST_Slope'])

 
    input_data_array = input_data.values

    l
    prediction = model.predict(input_data_array)

    
    if prediction[0] == 1:
        print("The model predicts: Heart Disease")
    else:
        print("The model predicts: No Heart Disease")


predict_heart_disease()
