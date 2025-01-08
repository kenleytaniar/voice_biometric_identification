import os
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sounddevice as sd
from scipy.io.wavfile import write

# Rekam suara
def record_audio(output_path, duration=5):
    print(f"Recording for {duration} seconds...")
    fs = 44100  # Sampling rate
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(output_path, fs, audio)
    print(f"Recording saved to {output_path}")

# Ekstraksi fitur suara
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Load dataset
def load_dataset(dataset_path):
    features = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(label)
    
    return np.array(features), np.array(labels)

# Training model
def train_model(features, labels):
    from imblearn.over_sampling import SMOTE # Library untuk membantu oversampling dengan teknik SMOTE
    import pandas as pd
    import matplotlib.pyplot as plt
    smote = SMOTE(random_state=42) # Menginisiasi SMOTE dengan random stat
    features_res, labels_res = smote.fit_resample(features, labels) # Mengoversampling

    X_train, X_test, y_train, y_test = train_test_split(features_res, labels_res, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return model

# Simpan model
def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")

# Prediksi suara
def predict_voice(model_path, audio_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    feature = extract_features(audio_path)
    prediction = model.predict([feature])
    # Logika prediksi
    if prediction[0] == "user1":
        result = "User 1"
    elif prediction[0] == "user2":
        result = "User 2"
    else:
        result = "Unknown User"
    
    print(f"Predicted class: {result}")
    return result



