import librosa
import numpy as np
from music21 import pitch, stream, note
import crepe # CREPE(Convolutional Representation For Pitch Estimation)

# old method with librosa ==> not accurate with high pitch worser quality
# ===========================
# def extract_pitch(file_path):
#     y, sr = librosa.load(file_path)
#     stft_result = librosa.stft(y)
#     power_spectrum = np.abs(stft_result) ** 2
#     log_power_spectrum = librosa.power_to_db(power_spectrum)
#     pitches, magnitudes = librosa.piptrack(S=log_power_spectrum, sr=sr)
#     return pitches, magnitudes

# Old method 
# ===================
# def pitches_to_notes(pitches, magnitudes, hop_length=512, sr=22050):
#     threshold = np.max(magnitudes) / 5
#     pitches[magnitudes < threshold] = 0
#     pitch_values = np.argmax(pitches, axis=0)
#     note_stream = stream.Stream()

#     for p in pitch_values:
#         if p > 0:
#             freq = p * sr / hop_length
#             note_name = pitch.Pitch(freq).nameWithOctave
#             note_stream.append(note.Note(note_name))
#     return note_stream

# Main function
# file_path = './output/htdemucs/the-moon-song/vocals.wav'
# pitches, magnitudes = extract_pitch(file_path)
# note_stream = pitches_to_notes(pitches, magnitudes)

# print("Notes in the melody:")
# for note in note_stream:
#     print(note.nameWithOctave)


# new method with crepe 
def extract_pitch(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)
    return frequency, confidence



# New method
def pitches_to_notes(frequencies, confidences, threshold=0.5):
    note_stream = stream.Stream()

    for freq, conf in zip(frequencies, confidences):
        if conf > threshold:
            note_stream.append(note.Note(pitch.Pitch(freq).nameWithOctave))

    return note_stream



file_path = './output/htdemucs/the-moon-song/vocals.wav'
frequencies, confidences = extract_pitch(file_path)
note_stream = pitches_to_notes(frequencies, confidences)

note_list = []
print("Notes in the melody:")
for note in note_stream:
    note_list.append(note.nameWithOctave)

print(note_list)



