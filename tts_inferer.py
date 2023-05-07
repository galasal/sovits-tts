import torch
from pathlib import Path
from so_vits_svc_fork.inference.infer_tool import Svc
import numpy as np
import librosa
import gradio as gr
import webbrowser
import azure.cognitiveservices.speech as speechsdk
import os
from emotion_classifier import emotion_classifier
from lxml import etree

class tts_inferer:
    def __init__(self, svc_config_path, svc_model_path):
        #initialise sovits model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        svc_config_path = Path(svc_config_path)
        svc_model_path = Path(svc_model_path)
        self.svc = Svc(
            net_g_path=svc_model_path.as_posix(),
            config_path=svc_config_path.as_posix(),
            cluster_model_path=None,
            device=device,
        )

        #initialise azure model
        self.azure_temp_file = "tmp/tmp.wav"
        speech_key = os.environ.get('SPEECH_KEY')
        service_region = os.environ.get('SPEECH_REGION')
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)  
        self.azureSynthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        #initialise emotion classification
        self.classifier = emotion_classifier()

    #does azure tts. Saves result to temp file and also returns it
    def azure_infer(self, text, speaker=None, emotion=None, speed=None):
        self.tree = etree.parse("azure.xml")
        self.tree.find(".//{*}prosody").text = text
        if emotion is not None:
            self.tree.find(".//{*}express-as").set("style", f"{emotion}")
        if speed is not None:
            self.tree.find(".//{*}prosody").set("rate", f"{speed}")
        if speaker is not None:
            self.tree.find(".//{*}voice").set("name", speaker)
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

    def infer(self, text, speaker=None, speed=None):
        emotion = self.classifier.map_to_azure_emotion(text)
        raw_audio = self.azure_infer(text=text, speaker=speaker, speed=speed, emotion=emotion)
        audio = self.svc_infer(raw_audio)
        return audio, raw_audio

