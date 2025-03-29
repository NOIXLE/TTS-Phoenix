# Phoenix TTS

### NGL, it's embarassing I still have trouble with git-

<p>So... this is a very simple TTS program made in Python using PyQt6 for the UI, and the kokoro-onnx package for the voice generation. Oh yeah, it also uses the sounddevice module to play audio.</p>

<p>It requires the original ONNX package and the voices binary obtained from the original page:</p>

> [kokoro-onnx -  Github](https://github.com/thewh1teagle/kokoro-onnx?tab=readme-ov-file)

## Setup
1. Download the repository files
2. Download the files kokoro-v1.0.onnx, and voices-v1.0.bin and place them in the same directory. (from kokoro-onnx's repository)
3. Download the required python modules with the command:
```
python pip install kokoro-onnx sounddevice PyQt6
```
4. Run the main.py file (highly recommend to use a python virtual environment like uv)
