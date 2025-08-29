import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
import joblib

# === Step 1: Load and merge datasets manually ===
csv_path = r"data\MachineLearningCVE\CICIDS2017_Merged.csv"
df = pd.read_csv(csv_path, low_memory=False)

# === Step 2: Filter only BENIGN traffic ===
label_column = next((col for col in df.columns if 'label' in col.lower().strip()), None)
if label_column is None:
    raise Exception("❌ 'Label' column not found.")
print(f"[✔] Found label column: '{label_column}'")

df = df[df[label_column].str.strip() == 'BENIGN']
df.drop(columns=[label_column], inplace=True)

# === Step 3: Remove non-numeric columns, handle NaNs/infs ===
df = df.select_dtypes(include=[np.number])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ✅ Save expected feature list
os.makedirs("model", exist_ok=True)
with open("model/feature_columns.txt", "w") as f:
    f.write("\n".join(df.columns))

# === Step 4: Normalize ===
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
joblib.dump(scaler, "model/scaler.pkl")

# === Step 5: Train/test split ===
X_train, X_test = train_test_split(df_scaled, test_size=0.2, random_state=42)

# === Step 6: Build autoencoder ===
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# === Step 7: Train ===
history = autoencoder.fit(X_train, X_train,
                          epochs=20,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1)

# === Step 8: Save model ===
autoencoder.save("model/autoencoder_model.h5")
print("[✔] Model and scaler saved successfully.")

# === Step 9: Plot loss ===
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
