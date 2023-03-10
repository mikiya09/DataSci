import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)


import os 
import requests 
import shutil 

_SAMPLE_DIR = "_assets"

SAMPLE_WAV_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "steam.wav")
SAMPLE_MP3_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.mp3"
SAMPLE_MP3_PATH = os.path.join(_SAMPLE_DIR, "steam.mp3")
SAMPLE_WAV_SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SAMPLE_WAV_SPEECH_PATH = os.path.join(_SAMPLE_DIR, "speech.wav")
SAMPLE_NOISE_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/distractors/rm1/babb/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"
SAMPLE_NOISE_PATH = os.path.join(_SAMPLE_DIR, "bg.wav")

# extract and creating audio data
# =========================================================
# os.makedirs(_SAMPLE_DIR, exist_ok=True)
#
# def fetch_audio_file(url, path):
#   with open(path, 'wb') as file_:
#     file_.write(requests.get(url).content)
#
# fetch_audio_file(SAMPLE_WAV_URL, SAMPLE_WAV_PATH)
# fetch_audio_file(SAMPLE_MP3_URL, SAMPLE_MP3_PATH)
# fetch_audio_file(SAMPLE_WAV_SPEECH_URL, SAMPLE_WAV_SPEECH_PATH)
# fetch_audio_file(SAMPLE_NOISE_URL, SAMPLE_NOISE_PATH)
# =========================================================


# quering audio metadata from url file
# =========================================================
# with requests.get(SAMPLE_WAV_URL, stream=True) as response:
#     filedata = response.raw 
#     metadata = torchaudio.info(filedata,format='wav')
#     print(f"Fetchd {filedata.tell()} bytes.")
#
# print(metadata)
# =========================================================

# load audio file 
waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
print(waveform.shape)
print(sample_rate)



# Plot waveform and spectrogram
import matplotlib.pyplot as plt

def _plot(waveform, sample_rate, title):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    if title == "Waveform":
      axes[c].plot(time_axis, waveform[c], linewidth=1)
      axes[c].grid(True)
    else:
      axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
  figure.suptitle(title)
  plt.show()


def plot_waveform(waveform, sample_rate):
  _plot(waveform, sample_rate, title="Waveform")

def plot_specgram(waveform, sample_rate):
  _plot(waveform, sample_rate, title="Spectrogram")

plot_waveform(waveform, sample_rate)
plot_specgram(waveform, sample_rate)
