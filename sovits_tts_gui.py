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
from tts_inferer import tts_inferer

def sovits_tts_infer(text, speaker, speed):
    audio, raw_audio = inferer.infer(text, speaker, speed)
    return (inferer.svc.target_sample, audio), (inferer.svc.target_sample, raw_audio)

def tts_infer(text):
    audio = inferer.azure_infer(text)
    return (inferer.svc.target_sample, audio)

def sovits_infer(file_path):
    audio, sr = librosa.load(file_path, inferer.svc.target_sample)  
    audio = inferer.svc_infer(audio)
    return (inferer.svc.target_sample, audio)

inferer = tts_inferer("purin")

app = gr.Blocks()
with app:
    with gr.Tab("sovits tts"):
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
                btn.click(sovits_tts_infer,
                            inputs=[textbox, speakerbox, duration_slider,],
                            outputs=[audio_output, raw_output])
    with gr.Tab("azure only"):
        with gr.Row():
            with gr.Column():
                textbox = gr.TextArea(label="Text",
                                        placeholder="Type your sentence here",
                                        value="Hello", elem_id=f"tts-input")
            with gr.Column():
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!")
                btn.click(tts_infer,
                            inputs=[textbox],
                            outputs=[audio_output])
    with gr.Tab("sovits only"):
        with gr.Row():
            with gr.Column():
                textbox = gr.TextArea(label="File",
                                        placeholder="Type path to wav file here",
                                        value="F:/AIVoice/sovits-tts/tmp/tmp.wav", elem_id=f"tts-input")
            with gr.Column():
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!")
                btn.click(sovits_infer,
                            inputs=[textbox],
                            outputs=[audio_output])
                
webbrowser.open("http://127.0.0.1:7860")
app.launch(share=False)