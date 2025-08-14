import os
import logging
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from yt_dlp import YoutubeDL
import whisper

MAX_VIDEOS = 100
TRANSCRIPTS_DIR = 'transcripts'
AUDIO_DIR = 'audio'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('youtube_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('YouTubeProcessor')

model = whisper.load_model("base")

def resolve_channel_url(input_url):
    if input_url.startswith("@"):
        input_url = f"https://www.youtube.com/{input_url}"

    ydl_opts = {'quiet': True, 'extract_flat': True, 'force_generic_extractor': True}

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(input_url, download=False)
            if 'channel_url' in info:
                return info['channel_url']
            elif 'uploader_url' in info and '/channel/' in info['uploader_url']:
                return info['uploader_url']
            elif 'webpage_url' in info:
                return info['webpage_url']
    except Exception as e:
        logger.error(f"Error resolving channel URL: {e}")
    return None

def fetch_channel_videos(channel_url):
    ydl_opts = {
        'quiet': True, 'extract_flat': True, 'skip_download': True,
        'playlistend': MAX_VIDEOS, 'ignoreerrors': True
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"{channel_url}/videos", download=False)
            return info.get('entries', [])
    except Exception as e:
        logger.error(f"Error fetching videos: {e}")
        return []

def download_audio(video_url, video_id):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}.mp3")
    if os.path.exists(audio_path):
        logger.info(f"Audio already exists for {video_id}, skipping download.")
        return audio_path

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(AUDIO_DIR, f"{video_id}.%(ext)s"),
        'quiet': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        if os.path.exists(audio_path):
            logger.info(f"Audio saved: {audio_path}")
            return audio_path
        else:
            logger.error(f"Failed to find audio after download for {video_id}")
            return None
    except Exception as e:
        logger.error(f"Audio download failed for {video_id}: {e}")
        return None

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.info(f"No YouTube transcript for video {video_id}")
        return None
    except Exception as e:
        logger.error(f"Error fetching transcript for {video_id}: {e}")
        return None

def save_transcript_text(transcript, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in transcript:
                f.write(entry['text'] + '\n')
        logger.info(f"Transcript saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving transcript to {filepath}: {e}")
        return False

def transcribe_audio_file(audio_path, transcript_path):
    try:
        logger.info(f"Transcribing audio with Whisper: {audio_path}")
        result = model.transcribe(audio_path)
        text = result['text']
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Whisper transcription saved: {transcript_path}")
        return True
    except Exception as e:
        logger.error(f"Error transcribing audio {audio_path}: {e}")
        return False

def process_channel(channel_input):
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    channel_url = resolve_channel_url(channel_input)
    if not channel_url:
        return {"error": "Failed to resolve channel URL"}

    logger.info(f"Resolved channel URL: {channel_url}")
    videos = fetch_channel_videos(channel_url)
    if not videos:
        return {"error": "No videos found"}

    summary_path = 'video_summary.txt'
    processed_videos = []

    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        for video in videos[:MAX_VIDEOS]:
            video_id = video.get('id')
            video_title = video.get('title', 'Unknown Title')
            upload_date = video.get('upload_date', 'Unknown Date')
            try:
                if upload_date != 'Unknown Date':
                    upload_date = datetime.strptime(upload_date, '%Y%m%d').strftime('%Y-%m-%d')
            except Exception:
                pass

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Processing video: {video_title} ({upload_date})")
            summary_file.write(f"{video_title} ({upload_date})\n")

            audio_path = download_audio(video_url, video_id)
            transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.txt")

            transcript = get_youtube_transcript(video_id)
            if transcript:
                save_transcript_text(transcript, transcript_path)
            elif audio_path:
                transcribe_audio_file(audio_path, transcript_path)
            else:
                logger.warning(f"Skipping transcription for {video_id} due to no audio.")

            processed_videos.append({"id": video_id, "title": video_title, "date": upload_date})

    return {"status": "done", "processed_videos": processed_videos}
