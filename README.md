# Music Genre Prediction
Using AI/ML Techniques to predict genre of music or song.

Link to Dataset [GTZAN Dataset] (https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

Made a Web App using Streamlit using DNN (Deep Neural Network) Model and deployed it on Streamlit Cloud [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sanjankalsi-music-genre-predictio-music-genre-prediction-n11yqc.streamlit.app/)

The Dataset contains 10 Genres of music :
## Genres
* Blues
* Classical
* Country
* Disco
* Hip-Hop
* Jazz
* Metal
* Pop
* Reggae
* Rock

## Dataset
* Each Genre contains 100 audio files each of duration 30s.
* 30s audio files are divided into 10 files of 3s each.

## Model
I have explored the problem using Convolutional Neural Network (CNN) Model which made use of features such as chroma STFT, spectral centroids, rms, and other extracted features that are present in features_3sec.csv file in the dataset.

## Results
![Training Loss Plot](https://github.com/SanjanKalsi/music-genre-prediction/blob/main/loss_and_accuracy.png)

This is the training/validation and test/validation accuracy obtained after 500 epochs.
