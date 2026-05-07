"""
Speech-to-Text Engine
Uses OpenAI Whisper
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


class SpeechToTextEngine:

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "float16",
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        if self.verbose:
            print(f"[STT] Loading OpenAI Whisper model: {model_size}")

        # Load whisper model
        self.model = whisper.load_model(model_size)

        if self.verbose:
            print("[STT] Model loaded ✅")

    def record_audio(self,duration: int = 5,samplerate: int = 16000,channels: int = 1,) -> np.ndarray:

        if self.verbose:
            print(f"[STT] Recording {duration} seconds of audio...")

        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
        )
        sd.wait()

        if self.verbose:
            print("[STT] Recording finished ✅")

        # Convert to mono float32 numpy array
        if channels > 1:
            audio = np.mean(audio, axis=1)

        return audio.squeeze()

    def transcribe_array(self, audio: np.ndarray, samplerate: int = 16000) -> str:

        if audio is None or len(audio) == 0:
            if self.verbose:
                print("[STT] Empty audio, nothing to transcribe.")
            return ""

        if self.verbose:
            print("[STT] Starting transcription...")

        # Save audio to temporary file 
        temp_file = "/tmp/whisper_audio.wav"
        sf.write(temp_file, audio, samplerate)

        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                temp_file,
                language="en",
                verbose=False,
            )

            transcript = result["text"].strip()

            if self.verbose:
                print(f"[STT] Transcription complete: \"{transcript}\"")

            return transcript
        finally:
            # Clean up temp file
            import os
            try:
                os.remove(temp_file)
            except:
                pass

    def record_and_transcribe(self,duration: int = 5,samplerate: int = 16000,) -> str:
        """Convenience method: record from mic + transcribe."""
        audio = self.record_audio(duration=duration, samplerate=samplerate)
        return self.transcribe_array(audio, samplerate=samplerate)


if __name__ == "__main__":
    # Simple test of STT engine
    stt = SpeechToTextEngine(verbose=True, model_size="tiny")

    print("\n" + "=" * 50)
    print("Testing Speech-to-Text Engine")
    print("=" * 50)
    print("\nSpeak now (5 seconds)...")

    text = stt.record_and_transcribe(duration=5)

    if text:
        print(f"\n✅ Transcription: {text}")
    else:
        print("\n❌ Failed to transcribe")