import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


model = joblib.load('heart_disease_model.pkl')

def predict_heart_disease():
    age = int(input("Enter age (in years): "))  
    sex = int(input("Enter sex (1 = Male, 0 = Female): "))  
    cp = int(input("Enter chest pain type (0 = no pain, 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain): "))
    trestbps = int(input("Enter resting blood pressure (in mmHg): "))  
    chol = int(input("Enter cholesterol level (in mg/dl): "))  
    fbs = int(input("Enter fasting blood sugar (1 = True, 0 = False): ")) 
    restecg = int(input("Enter resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy): "))
    thalach = int(input("Enter maximum heart rate achieved (in beats per minute): ")) 
    exang = int(input("Enter exercise-induced angina (1 = Yes, 0 = No): "))  
    oldpeak = float(input("Enter depression induced by exercise relative to rest (in depression units): "))  
    slope = int(input("Enter slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping): "))
    ca = int(input("Enter number of major vessels colored by fluoroscopy (0, 1, 2, or 3): ")) 
    thal = int(input("Enter thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect): ")) 

    
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        print("The model predicts: Heart Disease")
    else:
        print("The model predicts: No Heart Disease")

predict_heart_disease()
