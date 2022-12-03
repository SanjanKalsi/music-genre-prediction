import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras

import pydub
import librosa

st.title("Upload MP3 file")
st.markdown("---")

def extract_features_from_wav(wav_file_location):
    audio_sample_location_2 = wav_file_location
    data_2, sampling_rate_2 = librosa.load(audio_sample_location_2)
    print(data_2)
    #sampling rate fixed as the same in sample files
    sampling_rate_2 = 22050
    #sampling_rate_2 = 45600
    print(sampling_rate_2)


    length = data_2.size
    print("length: ", data_2.size)


    chroma_stft = librosa.feature.chroma_stft(y = data_2, sr = sampling_rate_2)
    chroma_stft_mean = chroma_stft.mean()
    chroma_stft_var = chroma_stft.var()
    print("chroma_stft | mean: ", chroma_stft_mean, " | var: ", chroma_stft_var)


    S, phase = librosa.magphase(librosa.stft(y = data_2))
    rms = librosa.feature.rms(S=S)
    rms_mean = rms.mean()
    rms_var = rms.var()
    print("rms | mean: ", rms_mean, " | var: ", rms_var)


    spectral_centroid = librosa.feature.spectral_centroid(y = data_2, sr = sampling_rate_2)
    spectral_centroid_mean = spectral_centroid.mean()
    spectral_centroid_var = spectral_centroid.var()
    print("spectral_centroid | mean: ", spectral_centroid_mean, " | var: ", spectral_centroid_var)


    spectral_bandwidth = librosa.feature.spectral_bandwidth(y = data_2, sr = sampling_rate_2)
    spectral_bandwidth_mean = spectral_bandwidth.mean()
    spectral_bandwidth_var = spectral_bandwidth.var()
    print("spectral_bandwidth | mean: ", spectral_bandwidth_mean, " | var: ", spectral_bandwidth_var)


    rolloff = librosa.feature.spectral_rolloff(y = data_2, sr = sampling_rate_2)
    rolloff_mean = rolloff.mean()
    rolloff_var = rolloff.var()
    print("rolloff | mean: ", rolloff_mean, " | var: ", rolloff_var)


    zero_crossing_rate = librosa.feature.zero_crossing_rate(y = data_2)
    zero_crossing_rate_mean = zero_crossing_rate.mean()
    zero_crossing_rate_var = zero_crossing_rate.var()
    print("zero_crossing_rate | mean: ", zero_crossing_rate_mean, " | var: ", zero_crossing_rate_var)


    harmony = librosa.effects.harmonic(y = data_2)
    harmony_mean = harmony.mean()
    harmony_var = harmony.var()
    print("harmony | mean: ", harmony_mean, " | var: ", harmony_var)


    C = np.abs(librosa.cqt(y = data_2, sr = sampling_rate_2, fmin = librosa.note_to_hz('A1')))
    freqs = librosa.cqt_frequencies(C.shape[0], fmin = librosa.note_to_hz('A1'))
    perceptr = librosa.perceptual_weighting(C**2, freqs, ref = np.max)
    perceptr_mean = perceptr.mean()
    perceptr_var = perceptr.var()
    print("perceptr | mean: ", perceptr_mean, " | var: ", perceptr_var)


    hop_length = 512
    oenv = librosa.onset.onset_strength(y = data_2, sr = sampling_rate_2, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr = sampling_rate_2, hop_length=hop_length)
    # Compute global onset autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size = tempogram.shape[0])
    ac_global = librosa.util.normalize(ac_global)
    # Estimate the global tempo for display purposes
    tempo = librosa.beat.tempo(onset_envelope = oenv, sr = sampling_rate_2, hop_length=hop_length)[0]
    print("tempo: ", tempo)


    mfccs = librosa.feature.mfcc(y=data_2, sr=sampling_rate_2, n_mfcc = 20)

    mfcc_1_mean = mfccs[0].mean()
    mfcc_1_var = mfccs[0].var()
    print("mfcc_1 | mean: ", mfcc_1_mean, " | var: ", mfcc_1_var)

    mfcc_2_mean = mfccs[1].mean()
    mfcc_2_var = mfccs[1].var()
    print("mfcc_2 | mean: ", mfcc_2_mean, " | var: ", mfcc_2_var)

    mfcc_3_mean = mfccs[2].mean()
    mfcc_3_var = mfccs[2].var()
    print("mfcc_3 | mean: ", mfcc_3_mean, " | var: ", mfcc_3_var)

    mfcc_4_mean = mfccs[3].mean()
    mfcc_4_var = mfccs[3].var()
    print("mfcc_4 | mean: ", mfcc_4_mean, " | var: ", mfcc_4_var)

    mfcc_5_mean = mfccs[4].mean()
    mfcc_5_var = mfccs[4].var()
    print("mfcc_5 | mean: ", mfcc_5_mean, " | var: ", mfcc_5_var)

    mfcc_6_mean = mfccs[5].mean()
    mfcc_6_var = mfccs[5].var()
    print("mfcc_6 | mean: ", mfcc_6_mean, " | var: ", mfcc_6_var)

    mfcc_7_mean = mfccs[6].mean()
    mfcc_7_var = mfccs[6].var()
    print("mfcc_7 | mean: ", mfcc_7_mean, " | var: ", mfcc_7_var)

    mfcc_8_mean = mfccs[7].mean()
    mfcc_8_var = mfccs[7].var()
    print("mfcc_8 | mean: ", mfcc_8_mean, " | var: ", mfcc_8_var)

    mfcc_9_mean = mfccs[8].mean()
    mfcc_9_var = mfccs[8].var()
    print("mfcc_9 | mean: ", mfcc_9_mean, " | var: ", mfcc_9_var)

    mfcc_10_mean = mfccs[9].mean()
    mfcc_10_var = mfccs[9].var()
    print("mfcc_10 | mean: ", mfcc_10_mean, " | var: ", mfcc_10_var)

    mfcc_11_mean = mfccs[10].mean()
    mfcc_11_var = mfccs[10].var()
    print("mfcc_11 | mean: ", mfcc_11_mean, " | var: ", mfcc_11_var)

    mfcc_12_mean = mfccs[11].mean()
    mfcc_12_var = mfccs[11].var()
    print("mfcc_12 | mean: ", mfcc_12_mean, " | var: ", mfcc_12_var)

    mfcc_13_mean = mfccs[12].mean()
    mfcc_13_var = mfccs[12].var()
    print("mfcc_13 | mean: ", mfcc_13_mean, " | var: ", mfcc_13_var)

    mfcc_14_mean = mfccs[13].mean()
    mfcc_14_var = mfccs[13].var()
    print("mfcc_14 | mean: ", mfcc_14_mean, " | var: ", mfcc_14_var)

    mfcc_15_mean = mfccs[14].mean()
    mfcc_15_var = mfccs[14].var()
    print("mfcc_15 | mean: ", mfcc_15_mean, " | var: ", mfcc_15_var)

    mfcc_16_mean = mfccs[15].mean()
    mfcc_16_var = mfccs[15].var()
    print("mfcc_16 | mean: ", mfcc_16_mean, " | var: ", mfcc_16_var)

    mfcc_17_mean = mfccs[16].mean()
    mfcc_17_var = mfccs[16].var()
    print("mfcc_17 | mean: ", mfcc_17_mean, " | var: ", mfcc_17_var)

    mfcc_18_mean = mfccs[17].mean()
    mfcc_18_var = mfccs[17].var()
    print("mfcc_18 | mean: ", mfcc_18_mean, " | var: ", mfcc_18_var)

    mfcc_19_mean = mfccs[18].mean()
    mfcc_19_var = mfccs[18].var()
    print("mfcc_19 | mean: ", mfcc_19_mean, " | var: ", mfcc_19_var)

    mfcc_20_mean = mfccs[19].mean()
    mfcc_20_var = mfccs[19].var()
    print("mfcc_20 | mean: ", mfcc_20_mean, " | var: ", mfcc_20_var)


    df_2 = pd.DataFrame({
        'length': [length],
        'chroma_stft_mean': [chroma_stft_mean.astype(np.float64)],
        'chroma_stft_var': [chroma_stft_var.astype(np.float64)],
        'rms_mean': [rms_mean.astype(np.float64)],
        'rms_var': [rms_var.astype(np.float64)],
        'spectral_centroid_mean': [spectral_centroid_mean.astype(np.float64)],
        'spectral_centroid_var': [spectral_centroid_var.astype(np.float64)],
        'spectral_bandwidth_mean': [spectral_bandwidth_mean.astype(np.float64)],
        'spectral_bandwidth_var': [spectral_bandwidth_var.astype(np.float64)],
        'rolloff_mean': [zero_crossing_rate_mean.astype(np.float64)],
        'rolloff_var': [zero_crossing_rate_var.astype(np.float64)],
        'zero_crossing_rate_mean': [zero_crossing_rate_mean.astype(np.float64)],
        'zero_crossing_rate_var': [zero_crossing_rate_var.astype(np.float64)],
        'harmony_mean': [harmony_mean.astype(np.float64)],
        'harmony_var': [harmony_var.astype(np.float64)],
        'perceptr_mean': [perceptr_mean.astype(np.float64)],
        'perceptr_var': [perceptr_var.astype(np.float64)],
        'tempo': [tempo.astype(np.float64)],
        'mfcc1_mean': [mfcc_1_mean.astype(np.float64)],
        'mfcc1_var': [mfcc_1_var.astype(np.float64)],
        'mfcc2_mean': [mfcc_2_mean.astype(np.float64)],
        'mfcc2_var': [mfcc_2_var.astype(np.float64)],
        'mfcc3_mean': [mfcc_3_mean.astype(np.float64)],
        'mfcc3_var': [mfcc_3_var.astype(np.float64)],
        'mfcc4_mean': [mfcc_4_mean.astype(np.float64)],
        'mfcc4_var': [mfcc_4_var.astype(np.float64)],
        'mfcc5_mean': [mfcc_5_mean.astype(np.float64)],
        'mfcc5_var': [mfcc_5_var.astype(np.float64)],
        'mfcc6_mean': [mfcc_6_mean.astype(np.float64)],
        'mfcc6_var': [mfcc_6_var.astype(np.float64)],
        'mfcc7_mean': [mfcc_7_mean.astype(np.float64)],
        'mfcc7_var': [mfcc_7_var.astype(np.float64)],
        'mfcc8_mean': [mfcc_8_mean.astype(np.float64)],
        'mfcc8_var': [mfcc_8_var.astype(np.float64)],
        'mfcc9_mean': [mfcc_9_mean.astype(np.float64)],
        'mfcc9_var': [mfcc_9_var.astype(np.float64)],
        'mfcc10_mean': [mfcc_10_mean.astype(np.float64)],
        'mfcc10_var': [mfcc_10_var.astype(np.float64)],
        'mfcc11_mean': [mfcc_11_mean.astype(np.float64)],
        'mfcc11_var': [mfcc_11_var.astype(np.float64)],
        'mfcc12_mean': [mfcc_12_mean.astype(np.float64)],
        'mfcc12_var': [mfcc_12_var.astype(np.float64)],
        'mfcc13_mean': [mfcc_13_mean.astype(np.float64)],
        'mfcc13_var': [mfcc_13_var.astype(np.float64)],
        'mfcc14_mean': [mfcc_14_mean.astype(np.float64)],
        'mfcc14_var': [mfcc_14_var.astype(np.float64)],
        'mfcc15_mean': [mfcc_15_mean.astype(np.float64)],
        'mfcc15_var': [mfcc_15_var.astype(np.float64)],
        'mfcc16_mean': [mfcc_16_mean.astype(np.float64)],
        'mfcc16_var': [mfcc_16_var.astype(np.float64)],
        'mfcc17_mean': [mfcc_17_mean.astype(np.float64)],
        'mfcc17_var': [mfcc_17_var.astype(np.float64)],
        'mfcc18_mean': [mfcc_18_mean.astype(np.float64)],
        'mfcc18_var': [mfcc_18_var.astype(np.float64)],
        'mfcc19_mean': [mfcc_19_mean.astype(np.float64)],
        'mfcc19_var': [mfcc_19_var.astype(np.float64)],
        'mfcc20_mean': [mfcc_20_mean.astype(np.float64)],
        'mfcc20_var': [mfcc_20_var.astype(np.float64)],
    })

    print("df_2: ", df_2)

    return df_2




def get_prediction_from_saved_model(df):

    #get final prediction from the upload
    generes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    reconstructed_model = keras.models.load_model("genre_prediction_model.h5")
    prediction = reconstructed_model.predict(df)
    prediction_with_round_off = np.round(prediction, 2)[0]

    prediction_list = []
    for value in prediction_with_round_off:
        prediction_list.append(value)

    print(prediction_list)
    max_value = max(prediction_list)
    max_index = prediction_list.index(max_value)

    message = "Genre predictied for the input music is " + generes[max_index].capitalize() + " with percentage match of " + str(max_value * 100) + "%"
    print(message)

    return message




uploaded_sound_file = st.file_uploader("Upload an MP3 file to Predict it's Genre", type=["mp3", "wav"])

is_sound_file = 0
if uploaded_sound_file is not None:

    with st.spinner('Processing...'):
        if uploaded_sound_file.name.endswith('wav'):
            print("WAV file")
            audio = pydub.AudioSegment.from_wav(uploaded_sound_file)
            audio.export("music_file.wav",format="wav")
            is_sound_file = 1
        elif uploaded_sound_file.name.endswith('mp3'):
            print("MP3 file")
            audio = pydub.AudioSegment.from_mp3(uploaded_sound_file)
            audio.export("music_file.wav",format="wav")
            is_sound_file = 1

        if is_sound_file == 1:
            df = extract_features_from_wav("music_file.wav")
            message = get_prediction_from_saved_model(df)
            st.success(message)

