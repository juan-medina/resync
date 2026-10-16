# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import sys
import json
import os
import time
from pathlib import Path
from rapidfuzz import fuzz
from unidecode import unidecode
import re
from google import genai
from dotenv import load_dotenv
from tqdm import tqdm
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Setup standard logger
log_level_str = os.getenv("LOG_LEVEL", "ERROR").upper()
log_level = getattr(logging, log_level_str, logging.ERROR)
log_file = PROJECT_ROOT / "sync.log"

logging.basicConfig(
    filename=str(log_file),
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("resync")

# Silence noisy third-party HTTP loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip().strip('"').strip("'")

# Load delays from .env or default
try:
    RETRY_DELAYS = [int(x.strip()) for x in os.getenv("RETRY_DELAYS", "5,15,30").split(",") if x.strip()]
except ValueError:
    RETRY_DELAYS = [5, 15, 30]


def normalize(text: str) -> str:
    t = unidecode(text or "").lower()
    t = re.sub(r"(\w)'(\w)", r"\1\2", t)
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def ask_gemini_batch(batch_data):
    if not API_KEY:
        logger.error("GEMINI_API_KEY is not set in .env")
        print("Error: GEMINI_API_KEY is not set in .env", file=sys.stderr)
        sys.exit(1)

    prompt_path = Path(__file__).resolve().parent / "labeler_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found at {prompt_path}")
        print(f"Error: Prompt file not found at {prompt_path}", file=sys.stderr)
        sys.exit(1)

    prompt += json.dumps(batch_data, indent=2)

    client = genai.Client(api_key=API_KEY)

    for delay in RETRY_DELAYS:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            logger.warning(f"Gemini API request failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    logger.error("All Gemini API retries exhausted for batch.")
    return {}


def main():
    try:
        dataset_path = PROJECT_ROOT / "data" / "raw" / "dataset.jsonl"
        labels_path = PROJECT_ROOT / "data" / "raw" / "labels.json"
        labels_path.parent.mkdir(parents=True, exist_ok=True)

        if not dataset_path.exists():
            msg = f"Dataset file not found at {dataset_path}. Run 'make dump' first."
            logger.error(msg)
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)

        labels = {}
        if labels_path.exists():
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load existing labels from {labels_path}: {e}", exc_info=True)

        batch = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Labeling tracks", unit="track"):
            try:
                row = json.loads(line)
            except Exception as e:
                logger.error(f"Corrupt JSON line in {dataset_path}: {e}")
                continue

            sp_id = row["spotify_id"]
            if sp_id in labels:
                continue

            sp_title = normalize(row["spotify_title"])
            sp_dur = row["spotify_duration_seconds"]
            sp_artists = row["spotify_artists"]

            candidates = row.get("yt_songs", []) + row.get("yt_videos", [])
            if not candidates:
                labels[sp_id] = -1
                continue

            found_perfect = False
            for i, cand in enumerate(candidates):
                yt_title = normalize(cand.get("title", ""))
                yt_dur = cand.get("duration_seconds")
                if not yt_dur:
                    continue

                is_atv = cand.get("videoType") == "MUSIC_VIDEO_TYPE_ATV"
                title_sim = fuzz.ratio(sp_title, yt_title)
                dur_diff = abs(sp_dur - int(yt_dur))

                yt_artist_str = normalize(" ".join([a.get("name", "") for a in cand.get("artists", [])]))
                artist_match = any(normalize(a) in yt_artist_str for a in sp_artists)

                if is_atv and title_sim > 90 and dur_diff <= 2 and artist_match:
                    labels[sp_id] = i
                    found_perfect = True
                    break

            if not found_perfect:
                cand_list = []
                for i, cand in enumerate(candidates):
                    ctype = cand.get("resultType", "video").upper()
                    vtype = cand.get("videoType", "")
                    type_desc = (
                        "Official Audio (ATV)"
                        if vtype == "MUSIC_VIDEO_TYPE_ATV"
                        else (
                            "Official Video (OMV)"
                            if vtype == "MUSIC_VIDEO_TYPE_OMV"
                            else ("Fan/UGC Video" if vtype == "MUSIC_VIDEO_TYPE_UGC" else ctype)
                        )
                    )
                    ctitle = cand.get("title", "")
                    calbum = (
                        cand.get("album", {}).get("name")
                        if isinstance(cand.get("album"), dict)
                        else cand.get("album")
                    )
                    album_str = f" [Album: {calbum}]" if calbum else ""
                    cdur = cand.get("duration_seconds", 0)
                    cartists = [a.get("name", "") for a in cand.get("artists", [])]
                    diff = abs(sp_dur - (cdur or 0))
                    cand_list.append({
                        "index": i,
                        "type": type_desc,
                        "info": f"'{ctitle}' by {', '.join(cartists)}{album_str} ({cdur}s) [Diff: {diff}s]",
                    })

                sp_album = row.get("spotify_album")
                sp_album_str = f" [Album: {sp_album}]" if sp_album else ""
                batch.append({
                    "spotify_id": sp_id,
                    "spotify_info": f"'{row['spotify_title']}' by {', '.join(sp_artists)}{sp_album_str} ({sp_dur}s)",
                    "candidates": cand_list,
                })

            if len(batch) >= 15:
                res = ask_gemini_batch(batch)
                for k, v in res.items():
                    labels[k] = v
                batch = []

                with open(labels_path, "w", encoding="utf-8") as out:
                    json.dump(labels, out, indent=2)
                time.sleep(2)

        if batch:
            print(f"Sending final batch of {len(batch)} ambiguous tracks to Gemini...", flush=True)
            res = ask_gemini_batch(batch)
            for k, v in res.items():
                labels[k] = v
            with open(labels_path, "w", encoding="utf-8") as out:
                json.dump(labels, out, indent=2)

        print(f"Labeling complete! Total labels: {len(labels)}. Saved to {labels_path}", flush=True)

    except KeyboardInterrupt:
        logger.warning("Labeling interrupted by user (Ctrl+C). Progress saved.")
        print("\nOperation paused by user (Ctrl+C). Progress saved.")
    except Exception as e:
        logger.error(f"Fatal error in ai_labeler: {e}", exc_info=True)
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
