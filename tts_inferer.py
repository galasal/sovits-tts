from vits import utils
from vits.text.symbols import symbols
from vits.models import SynthesizerTrn
from vits.text import text_to_sequence
import vits.commons as commons
import torch
from pathlib import Path
from so_vits_svc_fork.inference.infer_tool import Svc
import numpy as np
import librosa
import soundfile
import gradio as gr
import webbrowser

class tts_inferer:
    def __init__(self, vits_config_path, vits_model_path, svc_config_path, svc_model_path):
        #initialise vits model
        self.vits_hps = utils.get_hparams_from_file(vits_config_path)
        self.vits_net_g = SynthesizerTrn(
            len(symbols),
            self.vits_hps.data.filter_length // 2 + 1,
            self.vits_hps.train.segment_size // self.vits_hps.data.hop_length,
            n_speakers=self.vits_hps.data.n_speakers,
            **self.vits_hps.model).cuda()
        _ = self.vits_net_g.eval()
        _ = utils.load_checkpoint(vits_model_path, self.vits_net_g, None)

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

    def __get_text(self, text):
        text_norm = text_to_sequence(text, self.vits_hps.data.text_cleaners)
        if self.vits_hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    #turn text into speech using vits
    def __vits_infer(self, text, speaker, speed):
        stn_tst = self.__get_text(text)
        length_scale = 1 / speed
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([speaker]).cuda()
            audio = self.vits_net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
        return audio
    
    #convert audio using so-vits-svc
    def __svc_infer(self, audio):
        audio = librosa.resample(audio, orig_sr=self.vits_hps.data.sampling_rate, target_sr=self.svc.target_sample)
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
        audio = self.__vits_infer(text, speaker, speed)
        soundfile.write("tts.wav", audio, self.vits_hps.data.sampling_rate)
        audio = self.__svc_infer(audio)
        return audio


inferer = tts_inferer(
    vits_config_path="F:/AIVoice/data/models/vits_original/vctk_base.json", 
    vits_model_path="F:/AIVoice/data/models/vits_original/pretrained_vctk.pth", 
    svc_config_path="F:/AIVoice/data/sovits models/purin/config.json", 
    svc_model_path="F:/AIVoice/data/sovits models/purin/G_8000.pth")

#audio = inferer.infer("hello, how are you. I think this sounds a little strage, don't you? Let's make it talk for even longer.", 1000, 1)
#soundfile.write("output.wav", audio, inferer.svc.target_sample)

def gui_infer(text, speed):
    audio = inferer.infer(text, 0, speed)
    return (inferer.svc.target_sample, audio)


app = gr.Blocks()
with app:
    with gr.Tab("Text-to-Speech"):
        with gr.Row():
            with gr.Column():
                textbox = gr.TextArea(label="Text",
                                        placeholder="Type your sentence here",
                                        value="Hello", elem_id=f"tts-input")
                # select character
                duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
                                            label='Speed')
            with gr.Column():
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!")
                btn.click(gui_infer,
                            inputs=[textbox, duration_slider,],
                            outputs=[audio_output])
webbrowser.open("http://127.0.0.1:7860")
app.launch(share=False)