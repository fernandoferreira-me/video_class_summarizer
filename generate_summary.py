"""Video Class Summarizer.

This script downloads a video (Google Drive or external URL), extracts its
audio, transcribes it using OpenAI's Whisper model, and summarizes the
content using OpenAI GPT-4-turbo.

Features:
- Automatic handling of Google Drive and external links
- Audio extraction with ffmpeg
- Whisper-based audio transcription
- Context-aware summary generation via GPT-4-turbo
- Temporary file handling (no local file clutter)
- Timeout for optional user-provided context

Usage:
    python generate_summary.py <video_url> --model <whisper_model>

Example:
    python generate_summary.py https://drive.google.com/yourfile --model base

Author:
    Fernando Ferreira
"""

import os
import re
import hashlib
import shutil
import whisper
import torch
import tempfile
import subprocess
import argparse
import warnings
import requests
import gdown
import signal

from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead"
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")
client = OpenAI(api_key=api_key)


def get_cache_key(identifier: str) -> str:
    """Return a short deterministic key for a URL or file path."""
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]


def ask_redo(phase: str) -> bool:
    """Ask the user whether to redo an already-completed phase.

    Args:
        phase (str): Human-readable phase name.

    Returns:
        bool: True if the user wants to redo the phase.
    """
    answer = input(f"🔄 {phase} already cached. Redo? [y/N]: ").strip().lower()
    return answer == "y"


def is_google_drive_url(url: str) -> bool:
    """Check if the URL is a Google Drive link.

    Args:
        url (str): URL to check.

    Returns:
        bool: True if it's a Google Drive URL, else False.
    """
    return "drive.google.com" in url


def convert_drive_url(url: str) -> str:
    """Convert a shareable Google Drive link into a direct download link.

    Args:
        url (str): Original Google Drive link.

    Returns:
        str: Direct download link.
    """
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?id={file_id}&export=download"
    return url


def download_video(url: str, dest_path: str) -> None:
    """Download a video file using gdown or requests.

    Args:
        url (str): Source URL.
        dest_path (str): Destination file path.

    Raises:
        Exception: If download fails or invalid content type.
    """
    if is_google_drive_url(url):
        print(f"📥 Downloading from Google Drive (via gdown): {url}")
        gdown.download(url, dest_path, quiet=False)
    else:
        print(f"🌐 Downloading from external URL: {url}")
        r = requests.get(url, stream=True)
        content_type = r.headers.get("Content-Type", "")
        if r.status_code != 200 or "html" in content_type:
            raise Exception(
                f"❌ Invalid file download. Content-Type: {content_type}"
            )
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("✅ Download complete.")


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract mono 16kHz audio from a video file.

    Args:
        video_path (str): Video file path.
        audio_path (str): Output audio file path.
    """
    print("🎧 Extracting audio with ffmpeg...")
    command = [
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)
    print("✅ Audio extracted.")


def transcribe(file_path: str, model_name: str = "base") -> str:
    """Transcribe an audio file using Whisper.

    Args:
        file_path (str): Path to audio file.
        model_name (str, optional): Whisper model name.

    Returns:
        str: Transcribed text.
    """
    print(f"🧠 Transcribing audio with Whisper ({model_name})...")
    model = whisper.load_model(model_name).to("cpu")
    result = model.transcribe(file_path, language="pt")
    return result["text"]


def generate_summary(transcription: str, context: str) -> str:
    """Generate a friendly summary based on transcription and context.

    Args:
        transcription (str): Full transcribed text.
        context (str): Additional context for GPT.

    Returns:
        str: Generated summary.
    """
    print("💬 Asking GPT-4-turbo for class summary...")
    prompt = (
        f"{context}\n\n"
        "Fazer um resumo dessa aula, em tópicos para a turma! "
        "Lembrar os tópicos abordados. "
        "Falar de forma amigável e encorajadora.\n\n"
        f"Transcrição:\n{transcription}"
    )
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def save_output(content: str, filename: str) -> None:
    """Save string content to a text file.

    Args:
        content (str): Text content.
        filename (str): Output filename.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"💾 Saved: {filename}")


def input_with_timeout(prompt: str, timeout: int = 60) -> Optional[str]:
    """Prompt user input with timeout.

    Args:
        prompt (str): Prompt message.
        timeout (int, optional): Timeout in seconds.

    Returns:
        Optional[str]: User input or None if timeout.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        user_input = input(prompt)
        signal.alarm(0)
        return user_input
    except TimeoutError:
        print("\n⏰ Timeout reached. Proceeding without context.")
        return None


def main() -> None:
    """Main workflow: download or use local video, transcribe, summarize and save outputs.

    Intermediate results (audio, transcript, summary) are cached in .cache/<hash>/
    keyed by the video URL or absolute path. Each phase prompts the user before
    being redone when a cached result already exists.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe and summarize a video class."
    )
    parser.add_argument(
        "--video_url",
        help="Video URL (Google Drive or Zoom link)"
    )
    parser.add_argument(
        "--video_path",
        help="Local path to video file (e.g., .mp4)"
    )
    parser.add_argument(
        "--model", default="base",
        help="Whisper model: tiny, base, small, medium, large"
    )
    parser.add_argument(
        "--instructions", default=None,
        help="Instructions for GPT-4-turbo summary (optional)"
    )
    args = parser.parse_args()

    if not args.video_url and not args.video_path:
        parser.error("You must provide either --video_url or --video_path.")

    # Derive a stable cache key from the canonical video identifier.
    identifier = (
        os.path.abspath(args.video_path) if args.video_path else args.video_url
    )
    cache_dir = os.path.join(".cache", get_cache_key(identifier))
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📂 Cache directory: {cache_dir}")

    cached_audio = os.path.join(cache_dir, "audio.wav")
    cached_transcript = os.path.join(cache_dir, "transcript.txt")
    cached_summary = os.path.join(cache_dir, "summary.txt")

    # ── Phase 1: Download + audio extraction ────────────────────────────────
    if os.path.exists(cached_audio) and not ask_redo("Audio extraction"):
        print("♻️  Using cached audio.")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_video_path = os.path.join(tmp, "video.mp4")
            if args.video_path:
                shutil.copy(args.video_path, tmp_video_path)
            else:
                url = (
                    convert_drive_url(args.video_url)
                    if is_google_drive_url(args.video_url)
                    else args.video_url
                )
                download_video(url, tmp_video_path)
            extract_audio(tmp_video_path, cached_audio)

    # ── Phase 2: Transcription ───────────────────────────────────────────────
    if os.path.exists(cached_transcript) and not ask_redo("Transcription"):
        print("♻️  Using cached transcript.")
        with open(cached_transcript, "r", encoding="utf-8") as f:
            transcript = f.read()
    else:
        transcript = transcribe(cached_audio, args.model)
        save_output(transcript, cached_transcript)

    # ── Phase 3: Summary generation ──────────────────────────────────────────
    redo_summary = (
        not os.path.exists(cached_summary) or ask_redo("Summary generation")
    )
    if redo_summary:
        if not (context := args.instructions):
            context = input_with_timeout(
                "📝 (Optional) Context for the summary (30s timeout):\n> "
            ) or "Resumo da aula."
        summary = generate_summary(transcript, context)
        save_output(summary, cached_summary)
    else:
        print("♻️  Using cached summary.")
        with open(cached_summary, "r", encoding="utf-8") as f:
            summary = f.read()

    # ── Final outputs ────────────────────────────────────────────────────────
    save_output(transcript, "transcript.txt")
    save_output(summary, "summary.txt")

    print("\n✅ Summary ready:\n")
    print(summary)

if __name__ == "__main__":
    main()
