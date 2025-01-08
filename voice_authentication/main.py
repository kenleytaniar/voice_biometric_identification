import os
import streamlit as st
import pickle
from voice_auth_utils import (
    record_audio,
    load_dataset,
    train_model,
    save_model,
    predict_voice
)

# Path default
BASE_DIR = "voice_authentication"
DATASET_PATH =  "voice_dataset"
MODEL_PATH ="models/voice_auth_model.pkl"
RECORDINGS_PATH = "recordings/test_audio.wav"

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model belum dilatih. Silakan gunakan mode 'train' terlebih dahulu.")
        return None
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)

# Fungsi Streamlit UI
def main():
    st.title("Voice Authentication System")
    st.write("Pilih mode operasi di bawah ini:")

    # Menu pilihan mode
    mode = st.radio(
        "Mode Operasi:",
        options=["Record", "Train", "Predict"],
        index=0
    )

    if mode == "Record":
        st.subheader("Rekam Audio")
        if st.button("Mulai Rekaman"):
            record_audio(RECORDINGS_PATH)
            st.success(f"Audio direkam dan disimpan di: {RECORDINGS_PATH}")

    elif mode == "Train":
        st.subheader("Pelatihan Model")
        if st.button("Mulai Pelatihan"):
            features, labels = load_dataset(DATASET_PATH)
            model = train_model(features, labels)
            save_model(model, MODEL_PATH)
            st.success("Model berhasil dilatih dan disimpan.")


    elif mode == "Predict":
        st.subheader("Prediksi Suara")
        
        # Pilihan mode input audio
        input_option = st.radio(
            "Pilih sumber audio:",
            options=["Unggah file audio", "Gunakan file rekaman default"],
            index=0
        )

        if input_option == "Unggah file audio":
            uploaded_file = st.file_uploader("Unggah file audio untuk prediksi", type=["wav", "mp3"])

            if uploaded_file is not None:
                temp_audio_path = "recordings/temp_audio.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.audio(uploaded_file, format="audio/wav")

                # Tombol untuk prediksi
                if st.button("Prediksi"):
                    model = load_model()
                    if model:
                        prediction = predict_voice(MODEL_PATH, temp_audio_path)
                        st.success(f"Hasil prediksi: {prediction}")

        elif input_option == "Gunakan file rekaman default":
            st.write(f"File rekaman default: {RECORDINGS_PATH}")
            
            # Tombol untuk prediksi
            if st.button("Prediksi Rekaman Default"):
                if os.path.exists(RECORDINGS_PATH):
                    st.audio(RECORDINGS_PATH, format="audio/wav")
                    model = load_model()
                    if model:
                        prediction = predict_voice(MODEL_PATH, RECORDINGS_PATH)
                        
                        st.success(f"Hasil prediksi: {prediction}")
                else:
                    st.error("File rekaman default tidak ditemukan. Silakan rekam suara terlebih dahulu.")


if __name__ == "__main__":
    main()
