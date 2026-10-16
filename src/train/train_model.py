# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from dotenv import load_dotenv

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.common.matcher import MatchingEngine, parse_title, split_artists
from src.common.models import SpotifyTrack


def main():
    try:
        dataset_path = PROJECT_ROOT / "data" / "raw" / "dataset.jsonl"
        labels_path = PROJECT_ROOT / "data" / "raw" / "labels.json"
        model_path = PROJECT_ROOT / "models" / "matcher_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if not dataset_path.exists():
            msg = f"Dataset file not found at {dataset_path}. Run 'make dump' first."
            logger.error(msg)
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)

        if not labels_path.exists():
            msg = f"Labels file not found at {labels_path}. Run 'make label' first."
            logger.error(msg)
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)

        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        engine = MatchingEngine()

        X = []
        y = []

        print("Scoring candidates through existing MatchingEngine...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception as e:
                    logger.error(f"Corrupt JSON line in {dataset_path}: {e}")
                    continue

                sp_id = row["spotify_id"]
                if sp_id not in labels:
                    continue

                chosen_idx = labels[sp_id]
                songs = row.get("yt_songs", [])
                videos = row.get("yt_videos", [])
                candidates = songs + videos

                track = SpotifyTrack(
                    id=sp_id,
                    title=row["spotify_title"],
                    artists=row["spotify_artists"],
                    duration_ms=row["spotify_duration_seconds"] * 1000,
                    album=row.get("spotify_album"),
                )

                sp_parsed, _ = parse_title(track.title)

                for i, yt_item in enumerate(candidates):
                    is_video = i >= len(songs)
                    pos_in_group = i if not is_video else (i - len(songs))
                    yt_dur = yt_item.get("duration_seconds")
                    if not yt_dur:
                        yt_dur = 0

                    raw_title = yt_item.get("title", "")
                    yt_parsed, _ = parse_title(raw_title)

                    diff = abs(int(yt_dur) - track.duration_seconds)

                    t_score = engine.title_score(sp_parsed, yt_parsed)
                    ft_score = engine.full_title_score(track.title, raw_title)
                    d_score = engine.duration_score(diff, track.duration_seconds)
                    r_score = engine.rank_score(pos_in_group)
                    v_score = engine.version_score(sp_parsed.tags, yt_parsed.tags)
                    is_live_mismatch = 1.0 if ("live" not in sp_parsed.tags and "live" in yt_parsed.tags) else 0.0

                    vtype = yt_item.get("videoType", "")
                    is_atv = 1.0 if vtype == "MUSIC_VIDEO_TYPE_ATV" else 0.0
                    is_omv = 1.0 if vtype == "MUSIC_VIDEO_TYPE_OMV" else 0.0
                    is_ugc = 1.0 if vtype == "MUSIC_VIDEO_TYPE_UGC" else 0.0

                    sp_album = track.album
                    has_sp_album = 1.0 if (sp_album and sp_album.strip()) else 0.0
                    yt_album = (
                        yt_item.get("album", {}).get("name")
                        if isinstance(yt_item.get("album"), dict)
                        else yt_item.get("album")
                    )
                    has_yt_album = 1.0 if (yt_album and str(yt_album).strip()) else 0.0

                    from rapidfuzz import fuzz
                    from src.common.matcher import normalize

                    album_sim = (
                        (fuzz.token_set_ratio(normalize(sp_album), normalize(str(yt_album))) / 100.0)
                        if (has_sp_album and has_yt_album)
                        else 0.0
                    )
                    album_in_title = 1.0 if (has_sp_album and normalize(sp_album) in normalize(raw_title)) else 0.0

                    features = [
                        t_score,
                        ft_score,
                        d_score,
                        r_score,
                        v_score,
                        is_live_mismatch,
                        1.0 if is_video else 0.0,
                        is_atv,
                        is_omv,
                        is_ugc,
                        has_sp_album,
                        has_yt_album,
                        album_sim,
                        album_in_title,
                    ]
                    is_match = 1 if i == chosen_idx else 0

                    X.append(features)
                    y.append(is_match)

        if not X:
            msg = "No training data pairs could be extracted from dataset and labels."
            logger.error(msg)
            print(f"Error: {msg}", file=sys.stderr)
            sys.exit(1)

        print(f"Total candidate pairs extracted: {len(X)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)

        print("\n--- Model Evaluation ---")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        feature_names = [
            "Title Score",
            "Full Title Score",
            "Duration Score",
            "Rank Score",
            "Version Score",
            "Is Live Mismatch",
            "Is Video",
            "Is ATV",
            "Is OMV",
            "Is UGC",
            "Has Spotify Album",
            "Has YT Album",
            "Album Sim",
            "Album In Title",
        ]
        importances = model.feature_importances_
        print("\n--- Feature Importances ---")
        for name, imp in zip(feature_names, importances):
            print(f"{name}: {imp:.4f}")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")

    except Exception as e:
        logger.error(f"Fatal error in train_model: {e}", exc_info=True)
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
