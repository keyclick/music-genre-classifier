import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model


@st.cache_resource  
def load_trained_model():
    return load_model("lstm_audio_classification_model.keras")

model = load_trained_model()


genre_mapping = {
    0: "blues", 1: "classical", 2: "country", 3: "disco",
    4: "hiphop", 5: "jazz", 6: "metal", 7: "pop",
    8: "reggae", 9: "rock"
}


def preprocess_audio(file_path, max_timesteps, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050) 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T 


    if mfcc.shape[0] < max_timesteps:
        mfcc = np.pad(mfcc, ((0, max_timesteps - mfcc.shape[0]), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_timesteps, :]

    return np.expand_dims(mfcc, axis=0)


st.title("Music Genre Classifier")
st.write("Upload a `.wav` file, and this app will predict the genre!")



uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
  
    with open("uploaded_song.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
   
    max_timesteps = 1320  
    preprocessed_song = preprocess_audio("uploaded_song.wav", max_timesteps)
    
 
    predicted_probabilities = model.predict(preprocessed_song)
    predicted_genre_index = np.argmax(predicted_probabilities)
    predicted_genre = genre_mapping[predicted_genre_index]
    
    st.success(f"The predicted genre is: **{predicted_genre}**")
