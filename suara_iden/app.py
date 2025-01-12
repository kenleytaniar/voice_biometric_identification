import streamlit as st
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa

# Fungsi untuk ekstraksi fitur
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Fungsi untuk melatih model
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model trained with accuracy: {acc * 100:.2f}%")
    return model

# Fungsi untuk menyimpan model
def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    st.success("Model saved successfully.")

# Fungsi untuk memuat model
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Fungsi untuk prediksi suara
def predict_voice(model, file_path):
    feature = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(feature)
    return prediction[0]

# Streamlit UI
def main():
    st.title("Voice Recognition App")
    model_path = "voice_model.pkl"
    dataset_path = "dataset"

    # Pilihan menu
    menu = ["Train Model", "Predict Voice"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Train Model":
        st.subheader("Train Model")
        st.info("Dataset harus memiliki dua folder: 'kenley' dan 'unknown'.")
        if st.button("Train"):
            features, labels = [], []
            for label in ["kenley", "unknown"]:
                label_path = os.path.join(dataset_path, label)
                if not os.path.exists(label_path):
                    st.error(f"Folder {label_path} tidak ditemukan.")
                    return
                for file in os.listdir(label_path):
                    file_path = os.path.join(label_path, file)
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(label)

            features = np.array(features)
            labels = np.array(labels)

            model = train_model(features, labels)
            save_model(model, model_path)

    elif choice == "Predict Voice":
        st.subheader("Predict Voice")
        uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])
        if uploaded_file is not None:
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            if os.path.exists(model_path):
                model = load_model(model_path)
                result = predict_voice(model, "temp.wav")
                st.success(f"Prediction: {result}")
            else:
                st.error("Model belum dilatih. Silakan latih model terlebih dahulu.")

if __name__ == "__main__":
    main()
