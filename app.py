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
input_features = ['age', 'resting_bp', 'cholesterol', 'max_heartrate', 'st_depress',
                   'sex', 'fasting_sugar', 'exercise_angina',
                   'chest_pain', 'st_slope',
                   'rest_ecg', 'thal', 'vessels']

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
