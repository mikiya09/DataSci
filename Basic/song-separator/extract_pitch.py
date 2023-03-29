import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_pitch(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Compute the short-time Fourier transform (STFT) of the audio
    stft_result = librosa.stft(y)

    # Compute the power spectrum and convert to dB scale
    power_spectrum = np.abs(stft_result) ** 2
    log_power_spectrum = librosa.power_to_db(power_spectrum)

    # Compute the pitch (fundamental frequency) using the autocorrelation method
    pitches, magnitudes = librosa.piptrack(S=log_power_spectrum, sr=sr)

    return pitches, magnitudes

def plot_pitches(pitches, magnitudes):
    # Discard low-energy pitch estimates
    threshold = np.max(magnitudes) / 5
    pitches[magnitudes < threshold] = 0

    # Find the most dominant frequency (pitch) in each frame
    pitch_values = np.argmax(pitches, axis=0)

    # Plot the pitch values
    plt.figure(figsize=(12, 4))
    plt.plot(pitch_values, color='b', linewidth=1)
    plt.xlabel('Time (frames)')
    plt.ylabel('Pitch (frequency bin)')
    plt.title('Pitch contour')
    plt.show()

file_path = './output/htdemucs/the-moon-song/vocals.wav'
pitches, magnitudes = extract_pitch(file_path)
plot_pitches(pitches, magnitudes)
