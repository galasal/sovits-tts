import librosa
import gradio as gr
import webbrowser
from tts_inferer import tts_inferer
import os as os

model_names = [name for name in os.listdir("./models/")]

def sovits_tts_infer(text, use_azure):
    audio, raw_audio = inferer.infer(text, use_azure)
    return (inferer.svc.target_sample, audio), (inferer.svc.target_sample, raw_audio)

def tts_infer(text):
    audio = inferer.azure_infer(text)
    return (inferer.svc.target_sample, audio)

def sovits_infer(file_path):
    audio, sr = librosa.load(file_path, inferer.svc.target_sample)  
    audio = inferer.svc_infer(audio)
    return (inferer.svc.target_sample, audio)

def update_inferer(dropdown_value):
    global inferer
    inferer = tts_inferer(dropdown_value)

app = gr.Blocks()
with app:
    use_azure_checkbox = gr.Checkbox(label="Use Azure")
    inferer_dropdown = gr.Dropdown(model_names, label="Inferer")
    with gr.Tab("sovits tts"):
        with gr.Row():
            with gr.Column():
                textbox = gr.TextArea(label="Text",
                                        placeholder="Type your sentence here",
                                        value="Hello", elem_id=f"tts-input")
            with gr.Column():
                raw_output = gr.Audio(label="Raw Audio", elem_id="tts-audio")
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!")
                btn.click(sovits_tts_infer,
                            inputs=[textbox, use_azure_checkbox],
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
    inferer_dropdown.change(update_inferer, inputs=[inferer_dropdown])

webbrowser.open("http://127.0.0.1:7860")
app.launch(share=False)