from flask import Flask, render_template, request
import os
import numpy as np
import joblib
from keras.models import load_model
from werkzeug.utils import secure_filename
from pcap_parser import extract_features_from_pcap

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and scaler
model = load_model('model/autoencoder_model.h5')
scaler = joblib.load('model/scaler.pkl')
with open("model/feature_columns.txt", "r") as f:
    expected_cols = f.read().splitlines()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['pcap']
    if not uploaded_file.filename.endswith(".pcap"):
        return "❌ Please upload a valid .pcap file."

    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(filepath)

    # Extract flow features from pcap
    df = extract_features_from_pcap(filepath)
    df = df.select_dtypes(include=[np.number])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Align with training features
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Scale and predict
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    mse = np.mean(np.power(df_scaled - preds, 2), axis=1)

    # DEBUG: print MSE values to adjust threshold
    print("MSE values:", mse)
    print("Mean MSE:", np.mean(mse))

    # Adjusted threshold for testing
    threshold = np.mean(mse) + np.std(mse)
    print("Dynamic threshold:", threshold)

    anomalies = mse > threshold
    num_anomalies = int(np.sum(anomalies))
    total_flows = len(mse)

    return render_template("result.html", total=total_flows, anomalies=num_anomalies)

if __name__ == '__main__':
    app.run(debug=True)
