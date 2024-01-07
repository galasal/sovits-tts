import librosa
import gradio as gr
import webbrowser
from tts_inferer import tts_inferer
import os as os
import soundfile as sf

model_names = [name for name in os.listdir("./models/")]

def sovits_tts_infer(text, use_azure):
    audio, raw_audio = inferer.infer(text, use_azure)
    return (inferer.svc.target_sample, audio), (inferer.svc.target_sample, raw_audio)

def tts_infer(text):
    audio = inferer.azure_infer(text)
    return (inferer.svc.target_sample, audio)

def sovits_infer(file_path):
    sr, audio = file_path
    sf.write(inferer.audio_temp_file, audio, sr)
    audio, sr = librosa.load(inferer.audio_temp_file, inferer.svc.target_sample)  
    audio = inferer.svc_infer(audio)
    return (inferer.svc.target_sample, audio)

def update_inferer(dropdown_value):
    global inferer
    inferer = tts_inferer(dropdown_value)
    return "Model " + dropdown_value + " loaded"

app = gr.Blocks()
with app:
    with gr.Row():
        with gr.Column():
            inferer_dropdown = gr.Dropdown(model_names, label="Inferer")
            use_azure_checkbox = gr.Checkbox(label="Use Azure")
        with gr.Column(scale=0.2):
            inferer_loading = gr.Textbox(interactive=False, show_label=False)
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
                #textbox = gr.TextArea(label="File",
                                        #placeholder="Type path to wav file here",
                                        #value="F:/AIVoice/sovits-tts/tmp/tmp.wav", elem_id=f"tts-input")
                upload_audio = gr.Audio(label='audio to convert', source='upload', interactive=True)
            with gr.Column():
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!")
                btn.click(sovits_infer,
                            inputs=[upload_audio],
                            #inputs=[textbox],
                            outputs=[audio_output])
    inferer_dropdown.change(update_inferer, inputs=[inferer_dropdown], outputs=[inferer_loading])

webbrowser.open("http://127.0.0.1:7860")
app.launch(share=False)