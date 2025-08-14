import os
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from yt_dlp import YoutubeDL
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import whisper
import requests
import subprocess
import uuid

app = FastAPI(title="YouTube → Transcribe → Translate (All-in-One)")

# -------- Settings via env vars (safe defaults) ----------
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium
M2M_MODEL_NAME = os.getenv("M2M_MODEL", "facebook/m2m100_418M")
DEFAULT_SRC_LANG = os.getenv("DEFAULT_SRC_LANG", "en")
DEFAULT_TGT_LANG = os.getenv("DEFAULT_TGT_LANG", "fr")

# Lazy-loaded models
whisper_model = None
translator_model = None
translator_tokenizer = None


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
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
def process(req: ProcessRequest):
    """Full pipeline: download/convert -> transcribe -> translate."""
    try:
        src = req.source_lang or DEFAULT_SRC_LANG
        tgt = req.target_lang or DEFAULT_TGT_LANG

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, f"{uuid.uuid4()}.mp3")

            # Case 1: YouTube or supported sites
            if "youtube.com" in req.url or "youtu.be" in req.url:
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
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
                    raw_path = ydl.prepare_filename(info)
                    if raw_path.endswith(".webm"):
                        audio_path = raw_path.replace(".webm", ".mp3")
                    elif raw_path.endswith(".m4a"):
                        audio_path = raw_path.replace(".m4a", ".mp3")
                    else:
                        audio_path = os.path.splitext(raw_path)[0] + ".mp3"

            # Case 2: Direct .mp3 / .mp4
            elif req.url.endswith((".mp3", ".mp4")):
                r = requests.get(req.url, stream=True)
                r.raise_for_status()
                temp_file = os.path.join(tmpdir, os.path.basename(req.url))
                with open(temp_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                if temp_file.endswith(".mp4"):
                    subprocess.run(
                        ["ffmpeg", "-i", temp_file, audio_path, "-y"],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    audio_path = temp_file
            else:
                raise HTTPException(status_code=400, detail="Unsupported URL format.")

            # Step 2: Transcribe
            model = load_whisper()
            result = model.transcribe(audio_path, language=None)
            transcript_text = result.get("text", "").strip()

            # Step 3: Translate
            translated_text = translate_text(transcript_text, src_lang=src, tgt_lang=tgt)

        return {
            "status": "success",
            "source_lang": src,
            "target_lang": tgt,
            "transcript": transcript_text,
            "translation": translated_text,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
