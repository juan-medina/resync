# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import sys
import json
import time
import random
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

import logging

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

from src.common.spotify import get_spotify_playlist
from src.common.matcher import parse_title, primary_artist_display, _parse_float_range, _parse_int_list, AliasManager
from ytmusicapi import YTMusic


def main():
    try:
        raw_url = os.getenv("TRAINING_SPOTIFY_PLAYLIST_ID", "").strip().strip('"').strip("'")
        if not raw_url:
            raw_url = os.getenv("SPOTIFY_PLAYLIST_ID", "").strip().strip('"').strip("'")
        if not raw_url:
            msg = "Neither TRAINING_SPOTIFY_PLAYLIST_ID nor SPOTIFY_PLAYLIST_ID found in .env"
            logger.error(msg)
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)

        playlist_id = raw_url.split("/")[-1].split("?")[0] if "spotify.com" in raw_url else raw_url

        SEARCH_PACING_RANGE = _parse_float_range(os.getenv("SEARCH_PACING_RANGE", ""), (2.5, 5.0))
        RETRY_DELAYS = _parse_int_list(os.getenv("RETRY_DELAYS", ""), (45, 90, 180))
        max_tracks = int(os.getenv("SPOTIFY_MAX_TRACKS", "0"))

        print(f"Loading Spotify playlist: {playlist_id}...")
        try:
            playlist = get_spotify_playlist(playlist_id, max_tracks=max_tracks if max_tracks > 0 else None)
        except Exception as e:
            logger.error(f"Failed to fetch Spotify playlist '{playlist_id}': {e}", exc_info=True)
            print(f"Error fetching Spotify playlist: {e}", file=sys.stderr)
            sys.exit(1)

        tracks = playlist.tracks
        print(f"Loaded {len(tracks)} tracks.")

        browser_json_path = PROJECT_ROOT / "browser.json"
        if browser_json_path.exists():
            yt = YTMusic(str(browser_json_path))
        else:
            logger.warning(f"browser.json not found at {browser_json_path}, using unauthenticated YTMusic client.")
            yt = YTMusic()

        alias_path = PROJECT_ROOT / "data" / "state" / "artist_aliases.json"
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        alias_manager = AliasManager(str(alias_path))

        out_file = PROJECT_ROOT / "data" / "raw" / "dataset.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        processed_ids = set()
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_ids.add(data["spotify_id"])
                    except Exception:
                        pass

        if processed_ids:
            print(f"Found {len(processed_ids)} already processed tracks in {out_file.name}. Resuming where left off...")

        def safe_search(query, result_filter):
            last_error = None
            for attempt in range(len(RETRY_DELAYS) + 1):
                try:
                    pacing = random.uniform(*SEARCH_PACING_RANGE)
                    time.sleep(pacing)
                    res = yt.search(query, filter=result_filter, limit=20)
                    return res or []
                except Exception as e:
                    last_error = e
                    if attempt < len(RETRY_DELAYS):
                        time.sleep(RETRY_DELAYS[attempt])
            msg = f"Search failed for '{query}' ({result_filter}): {last_error}"
            logger.error(msg, exc_info=True)
            tqdm.write(f"\n{msg}")
            return []

        print("\nStarting overnight fetch. Press Ctrl+C at any time to pause (it will safely resume on the next run).")

        for t in tqdm(tracks, desc="Dumping Dataset"):
            if t.id in processed_ids:
                continue

            parsed, _ = parse_title(t.title)
            primary = primary_artist_display(t)

            aliases = []
            if primary:
                try:
                    aliases = alias_manager.fetch_aliases(primary) or []
                except Exception as e:
                    logger.error(f"Error fetching aliases for '{primary}': {e}", exc_info=True)

            query = f"{primary} {parsed.core_display}".strip()

            songs = safe_search(query, "songs")
            videos = safe_search(query, "videos")

            row = {
                "spotify_id": t.id,
                "spotify_title": t.title,
                "spotify_artists": t.artists,
                "spotify_album": getattr(t, "album", None),
                "spotify_duration_seconds": t.duration_seconds,
                "mb_aliases": aliases,
                "query_used": query,
                "yt_songs": songs,
                "yt_videos": videos,
            }

            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        print(f"\nDump complete! Saved to {out_file}")

    except KeyboardInterrupt:
        logger.warning("Dataset dump interrupted by user (Ctrl+C).")
        print("\n\nOperation paused by user (Ctrl+C). Progress saved.")
    except Exception as e:
        logger.error(f"Unexpected fatal error in dump_dataset: {e}", exc_info=True)
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
