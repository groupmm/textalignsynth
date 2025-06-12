from contextlib import nullcontext

import numpy as np

from .utils import suppress_stdout


class TTSWrapper:
    def __init__(self, device="cpu", language="en", verbose=False):
        self.verbose = verbose

        with suppress_stdout() if not self.verbose else nullcontext():
            from TTS.api import TTS

            if language == "en":
                self.synthesizer = TTS(
                    model_name="tts_models/en/ljspeech/tacotron2-DDC_ph", progress_bar=True
                ).to(device)
                self.fs = 22050

            elif language == "de":
                self.synthesizer = TTS(
                    model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=True
                ).to(device)
                self.fs = 22050

    def __call__(self, text):
        with suppress_stdout() if not self.verbose else nullcontext():
            return np.array(self.synthesizer.tts(text=text))
