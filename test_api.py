import requests

# Alamat API kamu
url = 'http://localhost:5000/predict'

# Contoh input data (harus sesuai dengan fitur yang dibutuhkan)
data = {
    "features": {
        "age": 63,
        "trestbps": 145,
        "chol": 233,
        "thalch": 150,
        "oldpeak": 2.3,
        "sex": 1,
        "fbs": 1,
        "exang": 0,
        "cp": 3,
        "slope": 0,
        "restecg": 0,
        "thal": 1,
        "ca": 0
    }
}

# Kirim request POST ke API
response = requests.post(url, json=data)

# Tampilkan hasil prediksi
if response.status_code == 200:
    print("Prediksi:", response.json()['prediction'])
else:
    print("Error:", response.status_code, response.text)
