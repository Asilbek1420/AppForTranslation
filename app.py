import os
import tempfile
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from yt_dlp import YoutubeDL

# Lazy imports to speed cold start until first call
whisper_model = None
translator_model = None
translator_tokenizer = None

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import whisper

app = FastAPI(title="YouTube → Transcribe → Translate (All-in-One)")

# -------- Settings via env vars (safe defaults) ----------
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium
M2M_MODEL_NAME = os.getenv("M2M_MODEL", "facebook/m2m100_418M")
DEFAULT_SRC_LANG = os.getenv("DEFAULT_SRC_LANG", "en")
DEFAULT_TGT_LANG = os.getenv("DEFAULT_TGT_LANG", "fr")


def load_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    return whisper_model


def load_m2m():
    global translator_model, translator_tokenizer
    if translator_model is None or translator_tokenizer is None:
        translator_tokenizer = M2M100Tokenizer.from_pretrained(M2M_MODEL_NAME)
        translator_model = M2M100ForConditionalGeneration.from_pretrained(M2M_MODEL_NAME)
        translator_model.eval()
    return translator_model, translator_tokenizer


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    model, tokenizer = load_m2m()
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_new_tokens=1024,
        )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


class ProcessRequest(BaseModel):
    url: str
    source_lang: Optional[str] = None   # e.g., "en", "ru", "uz"
    target_lang: Optional[str] = None   # e.g., "fr", "ar", "de"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
def process(req: ProcessRequest):
    """Full pipeline: download -> transcribe -> translate."""
    src = req.source_lang or DEFAULT_SRC_LANG
    tgt = req.target_lang or DEFAULT_TGT_LANG

    # 1) Download YouTube audio to /tmp as mp3
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "/tmp/%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "noplaylist": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(req.url, download=True)
        # normalize output path to mp3
        raw_path = ydl.prepare_filename(info)
        if raw_path.endswith(".webm"):
            audio_path = raw_path.replace(".webm", ".mp3")
        elif raw_path.endswith(".m4a"):
            audio_path = raw_path.replace(".m4a", ".mp3")
        else:
            # fallback if already mp3 or other ext
            audio_path = os.path.splitext(raw_path)[0] + ".mp3"

    # 2) Transcribe with Whisper
    model = load_whisper()
    result = model.transcribe(audio_path, language=None)  # let Whisper detect language
    transcript_text = result.get("text", "").strip()

    # 3) Translate with M2M-100
    translated_text = translate_text(transcript_text, src_lang=src, tgt_lang=tgt)

    # Optional cleanup
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        pass

    return {
        "source_lang": src,
        "target_lang": tgt,
        "transcript": transcript_text,
        "translation": translated_text,
    }
