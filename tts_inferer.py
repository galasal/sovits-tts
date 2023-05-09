import torch
from pathlib import Path
from so_vits_svc_fork.inference.core import Svc
import numpy as np
import librosa
import azure.cognitiveservices.speech as speechsdk
import os
from emotion_classifier import emotion_classifier
from lxml import etree
import audio_processor
import json

class tts_inferer:
    def __init__(self, model_folder_name):
        self.model_folder = Path("models/" + model_folder_name)
        self.__initialise_sovits()
        self.__initialise_azure()
        self.classifier = emotion_classifier()

    def __initialise_sovits(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        svc_config_path = self.model_folder / "config.json"
        svc_model_path = next(iter(self.model_folder.glob("G_*.pth")), None)
        self.svc = Svc(
            net_g_path=svc_model_path.as_posix(),
            config_path=svc_config_path.as_posix(),
            cluster_model_path=None,
            device=device,
        )

    def __initialise_azure(self):
        self.azure_temp_file = "tmp/tmp.wav"
        speech_key = os.environ.get('SPEECH_KEY')
        service_region = os.environ.get('SPEECH_REGION')
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)  
        self.azureSynthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    def __initialise_tts_config(self):
        tts_config_path = self.model_folder / "tts-config.json"
        with open(tts_config_path) as f:
            tts_config = json.load(f)
        self.base_voice = tts_config.get("baseVoice")
        self.high_pass_cutoff_freq = tts_config.get("highPassCutoffFreq")
        self.low_pass_cutoff_freq = tts_config.get("lowPassCutoffFreq")
        self.azure_pitch = tts_config.get("azurePitch")
        self.pitch_semitones = tts_config.get("pitchSemitones")
        self.speed = tts_config.get("speed")

    #does azure tts. Saves result to temp file and also returns it
    def azure_infer(self, text, speaker=None, emotion=None, speed=None, pitch=None):
        self.tree = etree.parse("azure.xml")
        self.tree.find(".//{*}prosody").text = text
        if emotion is not None:
            self.tree.find(".//{*}express-as").set("style", f"{emotion}")
        if speed is not None:
            self.tree.find(".//{*}prosody").set("rate", f"{speed}")
        if speaker is not None:
            self.tree.find(".//{*}voice").set("name", speaker)
        if pitch is not None:
            self.tree.find(".//{*}prosody").set("pitch", pitch)
        ssml_string = etree.tostring(self.tree).decode('utf-8')
        result = self.azureSynthesizer.speak_ssml_async(ssml_string).get()
        stream = speechsdk.AudioDataStream(result)
        stream.save_to_wav_file(self.azure_temp_file)
        audio, sr = librosa.load(self.azure_temp_file, sr=self.svc.target_sample)  
        return audio 

    #convert audio using so-vits-svc
    #audio should already be at correct sample rate
    def svc_infer(self, audio):
        audio = self.svc.infer_silence(
            audio.astype(np.float32),
            speaker=0,
            transpose=0,
            auto_predict_f0=0,
            cluster_infer_ratio=0,
            noise_scale=0.4,
            f0_method="dio",
            db_thresh=-40,
            pad_seconds=0.5,
            chunk_seconds=0.5,
            absolute_thresh=False,
        )

        return audio

    def infer(self, text):
        self.__initialise_tts_config()
        emotion = self.classifier.map_to_azure_emotion(text)
        raw_audio = self.azure_infer(text=text, speaker=self.base_voice, speed=self.speed, emotion=emotion, pitch=self.azure_pitch)
        if self.high_pass_cutoff_freq > 0:
            raw_audio = audio_processor.filter_audio(audio=raw_audio, sr=self.svc.target_sample, filter_type="highpass", cutoff_freq=self.high_pass_cutoff_freq)
        if self.low_pass_cutoff_freq > 0:
            raw_audio = audio_processor.filter_audio(audio=raw_audio, sr=self.svc.target_sample, filter_type="lowpass", cutoff_freq=self.low_pass_cutoff_freq)
        if self.pitch_semitones != 0:
            raw_audio = audio_processor.shift_frequency(audio=raw_audio, sr=self.svc.target_sample, shift_semitones=self.pitch_semitones)

        raw_audio = audio_processor.shift_frequency(raw_audio, self.svc.target_sample, 5)
        audio = self.svc_infer(raw_audio)
        return audio, raw_audio

