from pydub import AudioSegment

def convert_mp3_to_wav(input_file, output_file):
    # Load the .mp3 file
    audio = AudioSegment.from_mp3(input_file)

    # Export the audio as a .wav file
    audio.export(output_file, format='wav')

# loop through a file 
input_file = './output/htdemucs/月亮代表我的心/drums.mp3'
output_file = './output/htdemucs/月亮代表我的心/drums.wav'

convert_mp3_to_wav(input_file, output_file)
