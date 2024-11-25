import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


model = joblib.load('heart_disease_model.pkl')

def predict_heart_disease():
    age = int(input("Enter age: "))
    sex = int(input("Enter sex (1 = Male, 0 = Female): "))
    cp = int(input("Enter chest pain type (0, 1, 2, or 3): "))
    trestbps = int(input("Enter resting blood pressure: "))
    chol = int(input("Enter cholesterol level: "))
    fbs = int(input("Enter fasting blood sugar (1 = True, 0 = False): "))
    restecg = int(input("Enter resting electrocardiographic results (0, 1, 2): "))
    thalach = int(input("Enter maximum heart rate achieved: "))
    exang = int(input("Enter exercise induced angina (1 = Yes, 0 = No): "))
    oldpeak = float(input("Enter depression induced by exercise relative to rest: "))
    slope = int(input("Enter slope of the peak exercise ST segment (0, 1, 2): "))
    ca = int(input("Enter number of major vessels (0, 1, 2, 3): "))
    thal = int(input("Enter thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect): "))
    
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        print("The model predicts: Heart Disease")
    else:
        print("The model predicts: No Heart Disease")

predict_heart_disease()
