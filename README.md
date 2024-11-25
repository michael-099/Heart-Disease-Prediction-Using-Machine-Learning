# Heart-Disease-Prediction-Using-Machine-Learning
Here's the meaning of each of the columns you mentioned:

1. `cp` (Chest Pain Type):
   - This feature represents the type of chest pain experienced by the patient. It is a categorical variable with values ranging from 0 to 3, representing different types of chest pain:
     - `0`: Typical angina – A type of chest pain related to heart problems.
     - `1`: Atypical angina – Chest pain that doesn't necessarily indicate heart disease.
     - `2`: Non-anginal pain – Pain in the chest that is not related to heart issues.
     - `3`: Asymptomatic – No chest pain, meaning the patient shows no symptoms of heart disease.

2. `trestbps` (Resting Blood Pressure):
   - This feature represents the patient's blood pressure measured at rest, in mm Hg (millimeters of mercury).
   - It indicates the force of blood against the walls of the arteries when the heart is at rest.

3. `chol` (Serum Cholesterol):
   - This is the cholesterol level in the patient's serum (blood plasma), measured in mg/dl (milligrams per deciliter).
   - High cholesterol can contribute to heart disease by causing plaque buildup in the arteries.

4. `fbs` (Fasting Blood Sugar):
   - This is a binary feature indicating whether the patient's fasting blood sugar level is greater than 120 mg/dl:
     - `0`: No, the fasting blood sugar level is 120 mg/dl or lower.
     - `1`: Yes, the fasting blood sugar level is greater than 120 mg/dl.
   - Elevated blood sugar levels can indicate the presence of diabetes, which is a risk factor for heart disease.

5. `restecg` (Resting Electrocardiographic Results):
   - This feature represents the results of an electrocardiogram (ECG) taken while the patient is at rest. It is a categorical variable with values ranging from 0 to 2:
     - `0`: Normal – No abnormalities detected in the heart's electrical activity.
     - `1`: Having ST-T wave abnormality – Indicating a possible heart issue such as ischemia.
     - `2`: Showing probable or definite left ventricular hypertrophy – Enlarged heart muscle, often a sign of heart disease.

6. `thalach` (Maximum Heart Rate Achieved):
   - This feature indicates the maximum heart rate achieved by the patient during an exercise test. It is measured in beats per minute (bpm).
   - A higher heart rate can indicate a better cardiovascular condition, while an abnormally low heart rate during exercise could indicate heart issues.

7. `exang` (Exercise-Induced Angina):
   - This is a binary feature indicating whether the patient experiences angina (chest pain) during exercise:
     - `0`: No angina during exercise.
     - `1`: Yes angina during exercise.
   - Exercise-induced angina is a common symptom of heart disease.

8. `oldpeak` (ST Depression Induced by Exercise Relative to Rest):
   - This feature measures the ST depression on an ECG during exercise, relative to the resting state.
   - ST depression can indicate a lack of blood flow to the heart and is often a sign of heart disease.
   - The higher the value of `oldpeak`, the more significant the ischemia.

9. `slope` (Slope of the Peak Exercise ST Segment):
   - This feature represents the slope of the ST segment on the ECG during exercise. It is a categorical variable with values ranging from 0 to 2:
     - `0`: Upsloping – Indicates normal exercise response.
     - `1`: Flat – Suggests possible abnormal response or ischemia.
     - `2`: Downsloping – Often associated with severe heart problems.

---

These features represent various physiological and diagnostic measures used to assess heart health. Together, they can help predict the likelihood of heart disease in patients.

