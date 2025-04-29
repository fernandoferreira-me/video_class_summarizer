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
        gdown.download(url, dest_path, quiet=False, fuzzy=True)
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
        model="gpt-4-1106-preview",
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


def input_with_timeout(prompt: str, timeout: int = 30) -> Optional[str]:
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
    """Main workflow: download, transcribe, summarize and save outputs."""
    parser = argparse.ArgumentParser(
        description="Transcribe and summarize a video class."
    )
    parser.add_argument(
        "video_url", help="Video URL (Google Drive or Zoom link)"
    )
    parser.add_argument(
        "--model", default="base",
        help="Whisper model: tiny, base, small, medium, large"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "video.mp4")
        audio_path = os.path.join(tmp, "audio.wav")

        url = convert_drive_url(args.video_url) if is_google_drive_url(
            args.video_url
        ) else args.video_url
        download_video(url, video_path)
        extract_audio(video_path, audio_path)

        context = input_with_timeout(
            "📝 (Optional) Context for the summary "
            "(30s timeout):\n> "
        ) or "Resumo da aula."

        transcript = transcribe(audio_path, args.model)
        summary = generate_summary(transcript, context)

        save_output(transcript, "transcript.txt")
        save_output(summary, "summary.txt")

        print("\n✅ Summary ready:\n")
        print(summary)


if __name__ == "__main__":
    main()
