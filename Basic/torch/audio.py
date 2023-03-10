
# Audio file formats 
# .mp3 
# .flac 
# .wav 

import wave 

# Audio signal parameters
# =======================
# 1) number of channels 
#       - mono:
#       - stereo: two independent channels, audio coming from two directions(双声道)
# 2) sample with: the number of bytes for each sample
# 3) framerate/sample_rate: the number of sample value in seconds
#       - 44,110 Hz (standard sample_rate for CD)
# 4) number of frames? 
# 5) values of a frame

# read binary "rb"
obj = wave.open("./assets/speech.wav", "rb")
print("Number of channels:", obj.getnchannels())
print("Sample width:", obj.getsampwidth())
print("Frame rate:", obj.getframerate())
print("Number of frames:", obj.getnframes())
print("parameters:", obj.getparams())

# time 
t_audio = obj.getnframes() / obj.getframerate()
print(t_audio)
