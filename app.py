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
input_features = {
    "age": "age",
    "resting_bp": "trestbps",
    "cholesterol": "chol",
    "max_heartrate": "thalch",
    "st_depress": "oldpeak",
    "sex": "sex",
    "fasting_sugar": "fbs",
    "exercise_angina": "exang",
    "chest_pain": "cp",
    "st_slope": "slope",
    "rest_ecg": "restecg",
    "thal": "thal",
    "vessels": "ca"
}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            form_data = {}
            for form_key, model_key in input_features.items():
                form_data[model_key] = float(request.form[form_key])

            input_df = pd.DataFrame([form_data])

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
