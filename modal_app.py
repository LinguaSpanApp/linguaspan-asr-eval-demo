"""Modal app for our own (non-production) ASR inference, independent of the
production API's per-user rate limits. Two classes:

  YorubaASR  -- one fixed model (LinguaSpanApp/Yoruba-model-prod-finetunning-aug28)
                baked in at deploy time. src/modal_client.py calls this.
  GenericASR -- takes any HF model repo ID at call time (backs the Streamlit
                UI's "paste a model URL" field). src/generic_modal_client.py
                calls this.

Deploy (or redeploy after edits):
    modal deploy modal_app.py

Reuses the workspace's existing `huggingface-secret` (same one production
uses) to pull private model repos -- no new credentials needed.
"""
import modal

MODEL_REPO = "LinguaSpanApp/Yoruba-model-prod-finetunning-aug28"

app = modal.App("linguaspan-asr-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.46,<5",
        "accelerate",
        "librosa",
        "soundfile",
        "numpy",
        "sentencepiece",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=120,
)
class YorubaASR:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_REPO,
            device=device,
            torch_dtype=torch.float16 if device == 0 else torch.float32,
            chunk_length_s=30,
        )

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> dict:
        import io
        import time
        import librosa

        start = time.monotonic()
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        audio_duration = len(audio) / 16000

        result = self.pipe(audio)
        inference_time = time.monotonic() - start

        return {
            "transcription": result["text"].strip(),
            "audio_duration": audio_duration,
            "inference_time": inference_time,
        }


@app.cls(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=120,
)
class GenericASR:
    """Backs the Streamlit UI's "paste any HF model URL" field. Unlike
    YorubaASR (one model baked in at deploy time), this loads whatever
    model_repo is passed at call time via transformers.pipeline(), which
    auto-detects the architecture (Whisper, Wav2Vec2, etc.) from the repo's
    own config -- so it isn't hardcoded to Whisper. Pipelines are cached per
    model_repo on the container instance so repeat calls against the same
    model within a warm container don't reload it."""

    @modal.enter()
    def setup(self):
        self._pipelines = {}

    @modal.method()
    def transcribe(self, model_repo: str, audio_bytes: bytes) -> dict:
        import io
        import time
        import librosa
        import torch
        from transformers import pipeline

        if model_repo not in self._pipelines:
            device = 0 if torch.cuda.is_available() else -1
            self._pipelines[model_repo] = pipeline(
                "automatic-speech-recognition",
                model=model_repo,
                device=device,
                torch_dtype=torch.float16 if device == 0 else torch.float32,
                chunk_length_s=30,
            )
        pipe = self._pipelines[model_repo]

        start = time.monotonic()
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        audio_duration = len(audio) / 16000

        result = pipe(audio)
        inference_time = time.monotonic() - start

        return {
            "transcription": result["text"].strip(),
            "audio_duration": audio_duration,
            "inference_time": inference_time,
        }
