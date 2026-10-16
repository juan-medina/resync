# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

import socket
import logging
import requests

socket.setdefaulttimeout(15)

# Setup normal application logging to a file
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

# Monkey patch requests to log HTTP requests and responses on DEBUG / ERROR
original_request = requests.Session.request


def logged_request(self, method, url, **kwargs):
    logger.debug(f"[HTTP Request] {method} {url} | kwargs: {kwargs}")
    t0 = time.time()
    try:
        response = original_request(self, method, url, **kwargs)
        elapsed = time.time() - t0
        logger.debug(f"[HTTP Response] {method} {url} -> Status: {response.status_code} | Time: {elapsed:.2f}s | Response length: {len(response.content)}")
        return response
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[HTTP ERROR] {method} {url} -> Failed after {elapsed:.2f}s | Error: {e}", exc_info=True)
        raise


requests.Session.request = logged_request

from src.common.spotify import get_spotify_playlist
from src.common.matcher import MatchingEngine, SearchError, MIN_CONFIDENCE_THRESHOLD
from src.common.models import SpotifyTrack
from src.sync.state import StateManager

MAX_CONSECUTIVE_SEARCH_ERRORS: int = 5


def print_stats(processed_count: int, matched_records: List[Dict[str, Any]], unmatched_records: List[SpotifyTrack], elapsed_time: float, error_records: List[SpotifyTrack]) -> None:
    matched_count = len(matched_records)
    match_rate = (matched_count / processed_count * 100) if processed_count > 0 else 0.0

    scores = [r["score"] for r in matched_records]
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    high_conf = sum(1 for s in scores if s >= 0.85)
    med_conf = sum(1 for s in scores if MIN_CONFIDENCE_THRESHOLD <= s < 0.85)

    diffs = [r.get("duration_diff", 0) for r in matched_records]
    exact_dur = sum(1 for d in diffs if d == 0)
    delta_1s = sum(1 for d in diffs if d == 1)
    delta_2s_plus = sum(1 for d in diffs if d >= 2)

    songs_count = sum(1 for r in matched_records if r.get("match_type", "song") == "song")
    videos_count = sum(1 for r in matched_records if r.get("match_type") == "video")

    print("\n" + "=" * 60)
    print("           PLAYLIST MATCHING CONFIDENCE REPORT")
    print("=" * 60)
    print(f"Tracks Processed:       {processed_count}")
    print(f"Successfully Matched:   {matched_count} ({match_rate:.1f}%)")
    print(f"Unmatched / Discarded:  {len(unmatched_records)}")
    print(f"Search Errors:          {len(error_records)} (retried on next run)")
    print(f"Average Confidence:     {avg_score * 100:.1f}%\n")

    print("Source Type Breakdown:")
    print(f"  • Official Songs:     {songs_count} ({(songs_count / matched_count * 100) if matched_count else 0:.1f}%)")
    print(f"  • Music Videos:       {videos_count} ({(videos_count / matched_count * 100) if matched_count else 0:.1f}%)")

    print("\nConfidence Breakdown:")
    print(f"  • High (>= 85%):      {high_conf} ({(high_conf / matched_count * 100) if matched_count else 0:.1f}%)")
    print(f"  • Moderate ({MIN_CONFIDENCE_THRESHOLD * 100:.0f}-84%):  {med_conf} ({(med_conf / matched_count * 100) if matched_count else 0:.1f}%)")

    print("\nDuration Accuracy:")
    print(f"  • Exact (0s diff):    {exact_dur} ({(exact_dur / matched_count * 100) if matched_count else 0:.1f}%)")
    print(f"  • 1s deviation:       {delta_1s} ({(delta_1s / matched_count * 100) if matched_count else 0:.1f}%)")
    print(f"  • 2-4s deviation:     {delta_2s_plus} ({(delta_2s_plus / matched_count * 100) if matched_count else 0:.1f}%)")

    if unmatched_records:
        print("\nSample Unmatched Tracks (up to 5):")
        for u in unmatched_records[:5]:
            print(f"  - {u.title} - {u.artist_display} ({u.duration_seconds}s)")

    if error_records:
        print("\nTracks With Search Errors (up to 5):")
        for u in error_records[:5]:
            print(f"  - {u.title} - {u.artist_display}")

    print(f"\nTime Elapsed: {elapsed_time:.1f}s (avg {elapsed_time / max(1, processed_count):.2f}s per track)")
    print("=" * 60)


def main() -> None:
    try:
        raw_playlist_id = os.getenv("SPOTIFY_PLAYLIST_ID", "").strip().strip('"').strip("'")
        if not raw_playlist_id:
            msg = "SPOTIFY_PLAYLIST_ID not found in .env file."
            logger.error(msg)
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)

        playlist_id = raw_playlist_id.split("/")[-1].split("?")[0] if "spotify.com" in raw_playlist_id else raw_playlist_id

        max_tracks_env = os.getenv("SPOTIFY_MAX_TRACKS", "").strip().strip('"').strip("'")
        max_tracks = int(max_tracks_env) if max_tracks_env else None

        print(f"Reading target playlist: {playlist_id}")
        try:
            playlist = get_spotify_playlist(playlist_id, max_tracks=max_tracks)
        except Exception as e:
            logger.error(f"Failed to fetch Spotify playlist '{playlist_id}': {e}", exc_info=True)
            print(f"Error fetching Spotify playlist: {e}", file=sys.stderr)
            sys.exit(1)

        total_tracks = len(playlist.tracks)
        print(f"\nPlaylist '{playlist.name}' ready ({total_tracks} tracks).")

        dry_run = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")

        yt_playlist_id = os.getenv("YOUTUBE_PLAYLIST_ID", "").strip().strip('"').strip("'")
        if yt_playlist_id and "list=" in yt_playlist_id:
            import urllib.parse
            parsed_qs = urllib.parse.parse_qs(urllib.parse.urlparse(yt_playlist_id).query)
            if "list" in parsed_qs:
                yt_playlist_id = parsed_qs["list"][0]

        yt_client = None
        if not dry_run:
            from src.common.youtube import YouTubeMusicClient
            has_auth = (PROJECT_ROOT / "browser.json").exists() or (PROJECT_ROOT / "oauth.json").exists()
            if not has_auth:
                print("\n[ERROR] YouTube Music authentication file not found (browser.json or oauth.json).")
                print("Run '.\\scripts\\setup_auth.bat' to authenticate before syncing.")
                sys.exit(1)

            print("Validating YouTube Music authentication and playlist write access...")
            try:
                yt_client = YouTubeMusicClient()
                yt_client.check_auth(yt_playlist_id)
                print("YouTube Music authentication verified.\n")
            except (PermissionError, RuntimeError) as e:
                logger.error(f"Pre-flight authentication check failed: {e}")
                print(f"\n[ERROR] {e}\n", file=sys.stderr)
                sys.exit(1)

        print("Starting YouTube Music matching (Press Ctrl+C at any time to pause and view report):\n")

        matcher = MatchingEngine()
        state = StateManager()

        matched_records: List[Dict[str, Any]] = []
        unmatched_records: List[SpotifyTrack] = []
        error_records: List[SpotifyTrack] = []
        consecutive_errors = 0
        aborted = False
        start_time = time.time()

        progress_bar = tqdm(
            playlist.tracks,
            desc="Matching on YT Music",
            unit="track",
            dynamic_ncols=True,
        )

        try:
            for idx, track in enumerate(progress_bar, start=1):
                cached = state.get(track.id)
                if cached and cached.get("matched"):
                    matched_records.append(cached)
                else:
                    try:
                        match = matcher.match(track)
                        consecutive_errors = 0
                    except SearchError as e:
                        consecutive_errors += 1
                        error_records.append(track)
                        state.set(track.id, {"matched": False, "spotify_id": track.id, "error": str(e)})
                        logger.error(f"[SEARCH ERROR] {track.title} - {track.artist_display}: {e}", exc_info=True)
                        tqdm.write(f"[SEARCH ERROR] {track.title} - {track.artist_display}: {e}")
                        tqdm.write("\nStopping: Search failed. Check network / browser.json, then re-run.")
                        aborted = True
                        break

                    if match.is_matched and match.youtube_track:
                        record = {
                            "matched": True,
                            "spotify_id": track.id,
                            "spotify_title": track.title,
                            "spotify_artist": track.artist_display,
                            "yt_id": match.youtube_track.video_id,
                            "yt_title": match.youtube_track.title,
                            "yt_artist": match.youtube_track.artist,
                            "score": match.confidence_score,
                            "duration_diff": match.duration_diff_seconds,
                            "match_type": match.youtube_track.match_type,
                            "breakdown": match.breakdown,
                        }
                        state.set(track.id, record)
                        matched_records.append(record)
                    else:
                        unmatched_records.append(track)
                        rejected = match.best_rejected
                        state.set(track.id, {
                            "matched": False,
                            "spotify_id": track.id,
                            "spotify_title": track.title,
                            "spotify_artist": track.artist_display,
                            "best_rejected_title": rejected.title if rejected else None,
                            "best_rejected_id": rejected.video_id if rejected else None,
                            "best_rejected_score": match.confidence_score,
                            "breakdown": match.breakdown,
                        })

                if idx % 25 == 0:
                    state.save()

                current_matched = len(matched_records)
                current_rate = (current_matched / idx) * 100
                scores = [r["score"] for r in matched_records]
                avg_score = (sum(scores) / len(scores) * 100) if scores else 0.0

                progress_bar.set_postfix({
                    "matched": f"{current_rate:.1f}%",
                    "avg_conf": f"{avg_score:.1f}%",
                })

        except KeyboardInterrupt:
            logger.warning("Matching paused by user (Ctrl+C).")
            print("\n\nOperation paused by user (Ctrl+C). Progress saved to data/state/sync_state.json.")
            progress_bar.close()
            state.save()
            elapsed = time.time() - start_time
            processed = len(matched_records) + len(unmatched_records) + len(error_records)
            print_stats(processed, matched_records, unmatched_records, elapsed, error_records)
            return
        finally:
            progress_bar.close()
            state.save()

        elapsed = time.time() - start_time
        processed = len(matched_records) + len(unmatched_records) + len(error_records)
        print_stats(processed, matched_records, unmatched_records, elapsed, error_records)

        if aborted:
            print("\nRun stopped because of repeated search errors. Playlist sync skipped.")
            return

        dry_run = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")
        if dry_run:
            print("\n[DRY RUN] DRY_RUN=true is active in .env.")
            print("Matching complete and cached. Skipping YouTube playlist sync.")
            return

        has_auth = (PROJECT_ROOT / "browser.json").exists() or (PROJECT_ROOT / "oauth.json").exists()
        if not has_auth:
            print("\nNext step: To create/sync this playlist on YouTube Music, authenticate first.")
            print("Run: .\\scripts\\setup_auth.bat")
            return

        video_ids: List[str] = []
        seen = set()
        duplicates_removed = 0
        for r in matched_records:
            vid = r.get("yt_id")
            if vid:
                if vid not in seen:
                    seen.add(vid)
                    video_ids.append(vid)
                else:
                    duplicates_removed += 1

        if duplicates_removed > 0:
            print(f"\n[INFO] Removed {duplicates_removed} duplicate YouTube tracks (multiple Spotify versions mapped to the same video).")

        if not video_ids:
            print("\nNo matched tracks available to sync.")
            return

        if not yt_client:
            from src.common.youtube import YouTubeMusicClient
            yt_client = YouTubeMusicClient()

        if not yt_playlist_id:
            print(f"\nCreating new YouTube Music playlist: '{playlist.name}' ({len(video_ids)} tracks)...")
            try:
                from ytmusicapi.exceptions import YTMusicServerError, YTMusicUserError
                yt_playlist_id = yt_client.create_playlist(
                    title=playlist.name,
                    description="Synced from Spotify via resync",
                    privacy_status="PRIVATE",
                    video_ids=video_ids,
                )

                print("\nSUCCESS! Playlist created on YouTube Music:")
                print(f"  Title: {playlist.name}")
                print(f"  URL:   https://music.youtube.com/playlist?list={yt_playlist_id}")
                print(f"  ID:    {yt_playlist_id}")
                print(f"\nTip: Save YOUTUBE_PLAYLIST_ID={yt_playlist_id} in your .env for future continuous syncs.")
            except YTMusicServerError as e:
                logger.error(f"YouTube Music server error during playlist creation: {e}", exc_info=True)
                if "401" in str(e) or "404" in str(e):
                    print(f"\n[ERROR] YouTube Music rejected the request: {e}")
                    print("Your browser authentication cookies have likely expired, or you don't own this playlist.")
                    print("Please run `.\\scripts\\setup_auth.bat` to refresh your browser.json file.")
                else:
                    print(f"\n[ERROR] YouTube Music API Error: {e}")
            except Exception as e:
                logger.error(f"Failed to create YouTube playlist: {e}", exc_info=True)
                print(f"\n[ERROR] Failed to create YouTube playlist: {e}")
        else:
            print(f"\nSyncing with existing YouTube Music playlist ({yt_playlist_id})...")
            try:
                from ytmusicapi.exceptions import YTMusicServerError, YTMusicUserError
                existing_tracks = yt_client.get_playlist_tracks(yt_playlist_id)
                import collections
                target_counts = collections.Counter(video_ids)
                existing_counts = collections.Counter()

                to_remove_items = []

                for track in existing_tracks:
                    vid = track.get("videoId")
                    if existing_counts[vid] < target_counts[vid]:
                        existing_counts[vid] += 1
                    else:
                        to_remove_items.append(track)

                to_add = []
                for vid, target_qty in target_counts.items():
                    existing_qty = existing_counts.get(vid, 0)
                    if target_qty > existing_qty:
                        to_add.extend([vid] * (target_qty - existing_qty))

                print("\nSync Report:")
                print(f"  • Current YouTube playlist tracks: {len(existing_tracks)}")
                print(f"  • Desired target tracks:           {len(video_ids)}")
                print(f"  • Tracks to REMOVE:                {len(to_remove_items)}")
                print(f"  • Tracks to ADD:                   {len(to_add)}")

                if to_remove_items:
                    yt_client.remove_tracks(yt_playlist_id, to_remove_items)

                if to_add:
                    yt_client.add_tracks(yt_playlist_id, to_add)

                if not to_remove_items and not to_add:
                    print("\nPlaylist is already completely up to date (0 changes).")
                else:
                    print("\nSync complete!")
            except (PermissionError, YTMusicUserError) as e:
                logger.error(f"YouTube Music authentication/permission error: {e}", exc_info=True)
                print(f"\n[ERROR] YouTube Music Authentication Error:")
                print("Your session in browser.json is expired or missing edit permissions for this playlist.")
                print("Please run `.\\scripts\\setup_auth.bat` to refresh your authentication.")
            except YTMusicServerError as e:
                logger.error(f"YouTube Music server error during sync: {e}", exc_info=True)
                if "401" in str(e) or "404" in str(e):
                    print(f"\n[ERROR] YouTube Music rejected the request: {e}")
                    print("Your browser authentication cookies have likely expired, or you don't own this playlist.")
                    print("Please run `.\\scripts\\setup_auth.bat` to refresh your browser.json file.")
                else:
                    print(f"\n[ERROR] YouTube Music API Error: {e}")
            except Exception as e:
                logger.error(f"Failed to sync to YouTube playlist: {e}", exc_info=True)
                print(f"\n[ERROR] Failed to sync to YouTube playlist: {e}")

    except Exception as e:
        logger.error(f"Fatal error in main sync: {e}", exc_info=True)
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
