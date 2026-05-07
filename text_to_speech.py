"""
Text-to-Speech Engine
Uses pyttsx3 for offline TTS
"""

import threading
import pyttsx3


class TextToSpeechEngine:
    """Converts text to speech using pyttsx3."""

    def __init__(
        self,
        rate: int = 180,
        volume: float = 1.0,
        preferred_voice_substring: str | None = None,
        verbose: bool = False,
    ):
        """
        preferred_voice_substring: e.g. "female", "english", "US", etc.
        If None, keeps default voice.
        """
        self.verbose = verbose
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        if preferred_voice_substring:
            self._select_voice(preferred_voice_substring)

        if self.verbose:
            print("[TTS] Engine initialized ✅")

        # Prevent overlapping speeches
        self._lock = threading.Lock()

    def _select_voice(self, substring: str): # Not incorporated in main but could serve as additional functionality
        """Try to set a voice whose name/id contains the given substring."""
        voices = self.engine.getProperty("voices")
        chosen = None
        for v in voices:
            desc = f"{v.name} ({v.id})"
            if substring.lower() in desc.lower():
                chosen = v
                break

        if chosen is not None:
            self.engine.setProperty("voice", chosen.id)
            if self.verbose:
                print(f"[TTS] Selected voice: {chosen.name} ({chosen.id})")
        else:
            if self.verbose:
                print(f"[TTS] No voice matched '{substring}', using default.")

    def list_voices(self): # Not incorporated in main but could serve as additional functionality
        """Print all available voices."""
        voices = self.engine.getProperty("voices")
        print("\nAvailable voices:")
        for i, v in enumerate(voices):
            print(f"  [{i}] {v.name} ({v.id}) - {v.languages} {v.gender if hasattr(v, 'gender') else ''}")

    def _speak_blocking(self, text: str):
        """Speak text in the same thread (blocking)."""
        if self.verbose:
            print(f"[TTS] Speaking: \"{text}\"")
        self.engine.say(text)
        self.engine.runAndWait()

    def speak(self, text: str, block: bool = False):
        """
        Speak the given text.
        If block=False, speech runs in a separate thread.
        """
        if not text:
            return

        if block:
            with self._lock:
                self._speak_blocking(text)
        else:
            # Non-blocking: run in separate thread
            t = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
            t.start()


if __name__ == "__main__":
    tts = TextToSpeechEngine(verbose=True)

    tts.list_voices()

    samples = [
        "Hello! This is a test.",
        "This is the offline blind assistance device speaking.",
        "The system is now ready.",
    ]

    for s in samples:
        print(f"\nSpeaking: {s}")
        tts.speak(s, block=True)