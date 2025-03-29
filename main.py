import sys, os, json, re, queue, threading
import sounddevice as sd
import datetime
from PyQt6.QtWidgets import ( 
    QApplication,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QLabel,
    QTextEdit,
    QComboBox,
    QSlider,
    QPushButton
)
from PyQt6.QtCore import (
    Qt,
    QThread,
    pyqtSignal
)
import numpy as np
from kokoro_onnx import Kokoro

def get_model_path():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, "kokoro-v1.0.onnx")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "kokoro-v1.0.onnx")

LOG_FILE = "tts_log.txt"
VOICES_FILE = "voices-list.txt"
CONFIG_FILE = "config.json"

class TTSThread(QThread):
    audio_ready = pyqtSignal(np.ndarray, int)

    def __init__(self, kokoro, text, voice1, voice2, blend_value):
        super().__init__()
        self.kokoro = kokoro
        self.text = text
        self.voice1 = voice1
        self.voice2 = voice2
        self.blend_value = blend_value

    def run(self):
        try:
            blend = self.blend_value / 100
            blended_voice = (self.voice1 * blend) + (self.voice2 * (1 - blend))
            samples, sample_rate = self.kokoro.create(self.text, voice=blended_voice, speed=1.0, lang="en-us")
            self.audio_ready.emit(samples, sample_rate)
        except Exception as e:
            print(f"Error generating TTS: {e}")

class AudioThread(QThread):
    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.running = True

    def run(self):
        while self.running:
            try:
                samples, sample_rate = self.audio_queue.get(timeout=1)
                sd.play(samples, sample_rate)
                sd.wait()
                self.audio_queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def add_to_queue(self, samples, sample_rate):
        self.audio_queue.put((samples, sample_rate))

class TTSApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model_path = get_model_path()
        self.kokoro = None
        self.voices = self.load_voices()
        self.config = self.load_config()
        self.voice_cache = {}
        self.pinned = False
        self.init_ui()
        self.load_log()
        self.init_kokoro()
        self.audio_thread = AudioThread()
        self.audio_thread.start()

    def init_kokoro(self):
        """Initialize the TTS model in a separate thread to avoid blocking UI startup."""
        threading.Thread(target=self.load_kokoro, daemon=True).start()

    def load_kokoro(self):
        self.kokoro = Kokoro(self.model_path, "voices-v1.0.bin")

    def init_ui(self):
        self.setWindowTitle("Phoenix TTS")
        self.setGeometry(100, 100, 500, 300)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: white;
                font-size: 15px;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #1E1E1E;
                color: white;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 10px;
            }
            QSlider::groove:horizontal {
                height: 15px;
                background: #444;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: white;
                border-radius: 10px;
                width: 20px;
            }
        """)
        layout = QVBoxLayout()

        self.label = QLabel("Enter text:")
        self.text_input = QLineEdit()
        self.text_input.returnPressed.connect(self.handle_input)

        self.voice1_label = QLabel("Voice 1:")
        self.voice1_combo = QComboBox()
        self.voice2_label = QLabel("Voice 2:")
        self.voice2_combo = QComboBox()

        self.voice1_combo.addItems(self.voices)
        self.voice2_combo.addItems(self.voices)
        self.voice1_combo.setCurrentText(self.config.get("voice1", self.voices[0] if self.voices else ""))
        self.voice2_combo.setCurrentText(self.config.get("voice2", self.voices[1] if len(self.voices) > 1 else ""))
        self.voice1_combo.currentTextChanged.connect(self.save_config)
        self.voice2_combo.currentTextChanged.connect(self.save_config)

        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setMinimum(0)
        self.blend_slider.setMaximum(100)
        self.blend_slider.setValue(self.config.get("blend", 50))
        self.blend_slider.setTickInterval(5)
        self.blend_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.blend_value_label = QLabel(f"Blend: {self.blend_slider.value()}%")
        self.blend_slider.valueChanged.connect(self.update_blend_label)

        self.pin_button = QPushButton("Pin Window")
        self.pin_button.clicked.connect(self.toggle_pin)
        self.pin_button.setStyleSheet("padding: 5px; background-color: #333; color: white;")

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)

        layout.addWidget(self.label)
        layout.addWidget(self.text_input)
        layout.addWidget(self.voice1_label)
        layout.addWidget(self.voice1_combo)
        layout.addWidget(self.voice2_label)
        layout.addWidget(self.voice2_combo)
        layout.addWidget(self.blend_value_label)
        layout.addWidget(self.blend_slider)
        layout.addWidget(self.pin_button)
        layout.addWidget(self.log_display)
        self.setLayout(layout)

    def update_blend_label(self):
        self.blend_value_label.setText(f"Blend: {self.blend_slider.value()}%")
        self.save_config()

    def toggle_pin(self):
        self.pinned = not self.pinned
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self.pinned)
        self.pin_button.setText("Unpin Window" if self.pinned else "Pin Window")
        self.show()

    def handle_input(self):
        text = self.text_input.text().strip()
        text = re.sub(r'[^\w\s.,!?"\']', '', text)
        if text:
            self.text_input.clear()
            self.text_input.setFocus()
            self.log_text(text)
            self.play_audio(text)

    def play_audio(self, text):
        blend_value = self.blend_slider.value()
        voice1 = self.kokoro.get_voice_style(self.voice1_combo.currentText().strip())
        voice2 = self.kokoro.get_voice_style(self.voice2_combo.currentText().strip())
        self.tts_thread = TTSThread(self.kokoro, text, voice1, voice2, blend_value)
        self.tts_thread.audio_ready.connect(self.play_audio_output)
        self.tts_thread.start()

    def play_audio_output(self, samples, sample_rate):
        """Adds audio to the playback queue instead of interrupting."""
        self.audio_thread.add_to_queue(samples, sample_rate)

    def log_text(self, text):
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_entry = f"{timestamp} {text}"
        with open(LOG_FILE, "a", encoding="utf-8") as file:
            file.write(log_entry + "\n")
        self.log_display.append(log_entry)

    def load_log(self):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as file:
                last_lines = file.readlines()[-50:]
                self.log_display.setPlainText("".join(last_lines))
        except FileNotFoundError:
            self.log_display.setPlainText("No previous logs found.")

    def load_voices(self):
        try:
            with open(VOICES_FILE, "r", encoding="utf-8") as file:
                return [line.strip() for line in file.readlines() if line.strip()]
        except FileNotFoundError:
            return []

    def load_config(self):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_config(self):
        config_data = {
            "voice1": self.voice1_combo.currentText(),
            "voice2": self.voice2_combo.currentText(),
            "blend": self.blend_slider.value()
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as file:
            json.dump(config_data, file, indent=4)

    def closeEvent(self, event):
        """Ensure the audio thread stops when the UI closes."""
        self.audio_thread.stop()
        if hasattr(self, "tts_thread") and self.tts_thread.isRunning():
            self.tts_thread.quit()
            self.tts_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TTSApp()
    window.show()
    sys.exit(app.exec())