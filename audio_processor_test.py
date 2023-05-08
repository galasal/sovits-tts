import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import librosa
import soundfile
import scipy.signal as signal
import librosa
import audio_processor

file_path = "F:/AIVoice/sovits-tts/tmp/tmp.wav"
audio, sr = librosa.load(file_path, 44100)  

#audio = audio_processor.filter_audio(audio, sr, "highpass")
audio = audio_processor.shift_frequency(audio, sr, -5)


# Save the filtered audio to a file
soundfile.write("tmp/output.wav", audio, 44100)