# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import cog
import subprocess
import tempfile
from pathlib import Path

def generate_image_slides(audio_file, out_file):
    subprocess.call(
        [
            "python",
            "generate.py",
            "-p",
            "",
            "-ap",
            audio_file,
            "-o",
            out_file,
        ]
    )
    return out_file


class Predictor(cog.Predictor):
    @cog.input("audio", type=cog.Path, help="Input audio file")
    def predict(self, audio):
        """Run a single prediction on the model"""
        return generate_image_slides(audio, Path(tempfile.mkdtemp()) / "tmp.png")
