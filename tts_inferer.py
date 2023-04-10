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
        self.tree = etree.parse("azure.xml")
        speech_key = os.environ.get('SPEECH_KEY')
        service_region = os.environ.get('SPEECH_REGION')
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)  
        self.azureSynthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        #initialise emotion classification
        self.classifier = emotion_classifier()

    def __azure_infer(self, text, speaker, speed):
        emotion = self.classifier.map_to_azure_emotion(text)
        self.tree.find(".//{*}express-as").set("style", f"{emotion}")
        self.tree.find(".//{*}prosody").text = text
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
    def __svc_infer(self, audio):
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

    def infer(self, text, speaker, speed):
        raw_audio = self.__azure_infer(text, speaker, speed)
        audio = self.__svc_infer(raw_audio)
        return audio, raw_audio

def gui_inference(text, speaker, speed):
    audio, raw_audio = inferer.infer(text, speaker, speed)
    return (inferer.svc.target_sample, audio), (inferer.svc.target_sample, raw_audio)

inferer = tts_inferer(
    svc_config_path="F:/AIVoice/data/sovits models/purin/config.json", 
    svc_model_path="F:/AIVoice/data/sovits models/purin/G_8000.pth")

app = gr.Blocks()
with app:
    with gr.Tab("Text-to-Speech"):
        with gr.Row():
            with gr.Column():
                textbox = gr.TextArea(label="Text",
                                        placeholder="Type your sentence here",
                                        value="Hello", elem_id=f"tts-input")
                speakerbox = gr.Textbox(label="Text",
                                        placeholder="Speaker name here",
                                        value="en-US-JennyNeural", elem_id=f"speaker-input")
                # select character
                duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
                                            label='Speed')
            with gr.Column():
                raw_output = gr.Audio(label="Raw Audio", elem_id="tts-audio")
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!")
                btn.click(gui_inference,
                            inputs=[textbox, speakerbox, duration_slider,],
                            outputs=[audio_output, raw_output])
webbrowser.open("http://127.0.0.1:7860")
app.launch(share=False)