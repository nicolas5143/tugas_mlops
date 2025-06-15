from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model dan preprocessing tools
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
scaler = joblib.load('scaler.pkl')

# Fitur input
input_features = [
    'age', 'resting blood pressure (in mmHg)', 'cholesterol', 'maximum heart rate', 'ST depression induced by exercise relative to rest',
    'sex (Male=0, Female=1)', 'fasting blood sugar if fasting blood sugar > 120 mg/dl (true=0, false=1)', 'exercise include angina (true=0, false=1)',
    'chest pain (typical angina=0, atypicial angina=1, non-anginal=2, asymptomatic=3)', 'the slope of the peak exercise ST segment',
    'resting electrocardiographic results (normal=0, stt abnormality=1, lv hypertrophy=2)', 'thal (normal=0, fixed defect=1, reversible effect=2)', 'number of major vessels (0-3) colored by fluoroscopy'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Ambil data dari form
            features = [float(request.form[feat]) for feat in input_features]
            input_df = pd.DataFrame([dict(zip(input_features, features))])
            
            # Preprocessing dan prediksi
            X_preprocessed = preprocessor.transform(input_df)
            X_scaled = scaler.transform(X_preprocessed)
            result = model.predict(X_scaled)

            prediction = int(result[0])
        except Exception as e:
            prediction = f"Error: {e}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
