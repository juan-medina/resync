# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import math
import re
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz
from unidecode import unidecode
import requests
import json
import pickle
from ytmusicapi import YTMusic

from src.common.models import SpotifyTrack, TrackMatch, YouTubeTrack
from dotenv import load_dotenv
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_PATH = PROJECT_ROOT / "models" / "matcher_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        ML_MODEL = pickle.load(f)
except Exception:
    ML_MODEL = None

logger = logging.getLogger("resync")

# ---------------------------------------------------------------------------
# Scoring configuration (every signal is 0.0 - 1.0, weights sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS: Dict[str, float] = {
    "title": 0.35,
    "full_title": 0.15,
    "duration": 0.20,
    "rank": 0.15,
    "version": 0.10,
    "source": 0.05,
}
MIN_CONFIDENCE_THRESHOLD: float = 0.00
EARLY_STOP_THRESHOLD: float = 0.95

def _parse_float_range(env_val: str, default: Tuple[float, float]) -> Tuple[float, float]:
    if not env_val: return default
    parts = [x.strip() for x in env_val.split(",")]
    if len(parts) >= 2:
        try: return (float(parts[0]), float(parts[1]))
        except ValueError: pass
    return default

def _parse_int_list(env_val: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if not env_val: return default
    parts = [x.strip() for x in env_val.split(",") if x.strip()]
    try: return tuple(int(x) for x in parts) if parts else default
    except ValueError: return default

SEARCH_PACING_RANGE_SECONDS: Tuple[float, float] = _parse_float_range(os.getenv("SEARCH_PACING_RANGE", ""), (2.5, 5.0))
RETRY_DELAYS_SECONDS: Tuple[int, ...] = _parse_int_list(os.getenv("RETRY_DELAYS", ""), (45, 90, 180))
MAX_RESULTS_PER_SEARCH: int = 20

# ---------------------------------------------------------------------------
# Title vocabulary
# ---------------------------------------------------------------------------
# Words that describe a release/packaging, not the song. Dropped from titles.
NOISE_WORDS: Set[str] = {
    "remaster", "remastered", "remasterd", "master", "mastered", "rerecorded", "recorded",
    "version", "edit", "mono", "stereo", "digital", "clean", "explicit", "dirty",
    "single", "album", "original", "radio", "mix", "bonus", "track", "deluxe", "edition",
    "anniversary", "expanded", "reissue", "extended", "short", "full", "long", "main", "vocal",
    "official", "audio", "video", "music", "lyric", "lyrics", "visualizer", "hd", "hq", "4k",
    "from", "soundtrack", "ost", "theme", "motion", "picture", "film", "movie", "trailer",
    "reveal", "feat", "ft", "featuring",
}
# Version words: different recording of the same song.
LIVE_TAGS: Set[str] = {"live", "unplugged", "concert"}
ALT_TAGS: Set[str] = {"acoustic", "demo", "remix", "rmx", "orchestral", "piano", "slowed", "sped", "nightcore"}
BAD_TAGS: Set[str] = {"karaoke", "cover", "tribute", "instrumental", "reaction", "backing"}
VERSION_TAGS: Set[str] = LIVE_TAGS | ALT_TAGS | BAD_TAGS

FEATURE_PREFIXES: Set[str] = {"feat", "ft", "featuring", "with"}

SEGMENT_SPLIT = re.compile(r"\s+[-\u2013\u2014]\s+|[\(\)\[\]\{\};]|\s+/\s+")
ARTIST_SPLIT = re.compile(r"\s+(?:feat\.?|ft\.?|featuring|duet\s+with|with|&)\s+|\s*,\s*", re.IGNORECASE)


class SearchError(Exception):
    """Raised when YouTube Music search keeps failing after all retries."""


@dataclass
class ParsedTitle:
    """A title split into the song identity (core) and its qualifiers."""
    core_norm: str
    core_display: str
    tags: Set[str] = field(default_factory=set)


@dataclass
class Candidate:
    track: YouTubeTrack
    score: float
    duration_diff: int
    breakdown: Dict[str, float]


def normalize(text: str) -> str:
    """ASCII-fold, lowercase, join intra-word apostrophes/hyphens, strip punctuation."""
    t = unidecode(text or "").lower()
    t = re.sub(r"(\w)'(\w)", r"\1\2", t)
    t = re.sub(r"(\w)-(\w)", r"\1\2", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def _contains_words(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def split_artists(artists: List[str]) -> List[str]:
    """Expand 'A Duet with B' / 'A feat. B' / 'A & B' into individual normalized names."""
    names: List[str] = []
    for raw in artists:
        for piece in [raw] + ARTIST_SPLIT.split(raw):
            n = normalize(piece)
            if n and n not in names:
                names.append(n)
    return names


def primary_artist_display(track: SpotifyTrack) -> str:
    first = track.artists[0] if track.artists else ""
    return ARTIST_SPLIT.split(first)[0].strip()


def parse_title(title: str, strip_artists: Optional[List[str]] = None) -> Tuple[ParsedTitle, bool]:
    """Split a title into core song name + version tags.

    strip_artists: for videos, a segment that is just the artist name ('Artist - Song') is removed.
    Returns (parsed, artist_segment_removed).
    """
    raw_segments = [s.strip() for s in SEGMENT_SPLIT.split(title or "")]
    segments = [s for s in raw_segments if normalize(s)]

    artist_removed = False
    if strip_artists and len(segments) > 1:
        for i, seg in enumerate(segments):
            seg_norm = normalize(seg)
            if any(_contains_words(seg_norm, a) and len(seg_norm.split()) <= len(a.split()) + 3 for a in strip_artists):
                segments.pop(i)
                artist_removed = True
                break

    core_norm_parts: List[str] = []
    core_display_parts: List[str] = []
    tags: Set[str] = set()

    for idx, seg in enumerate(segments):
        tokens = normalize(seg).split()
        if idx == 0:
            core_norm_parts.append(" ".join(tokens))
            core_display_parts.append(seg)
            continue
        if tokens[0] in FEATURE_PREFIXES:
            continue
        is_qualifier = any(t in NOISE_WORDS or t in VERSION_TAGS or t.isdigit() for t in tokens)
        if is_qualifier:
            tags.update(t for t in tokens if t in VERSION_TAGS)
        else:
            core_norm_parts.append(" ".join(tokens))
            core_display_parts.append(seg)

    return ParsedTitle(" ".join(core_norm_parts), " ".join(core_display_parts), tags), artist_removed


class AliasManager:
    def __init__(self, cache_file=None):
        if cache_file is None:
            self.cache_file = PROJECT_ROOT / "data" / "state" / "artist_aliases.json"
        else:
            self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self):
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def fetch_aliases(self, artist_name: str) -> List[str]:
        key = artist_name.strip().lower()
        if key in self.cache:
            return self.cache[key]
            
        t0 = time.time()
        import os
        mb_ua = os.getenv("MB_USER_AGENT", "SpotifyYTMusicSync/1.0 (personal-script-resync@example.com)").strip('"').strip("'")
        headers = {"User-Agent": mb_ua}
        mbid = None
        
        # Load retry delays from env, fallback to default if missing
        import os
        retry_delays_str = os.getenv("RETRY_DELAYS", "")
        retry_delays = (15, 30, 45)
        if retry_delays_str:
            try:
                retry_delays = tuple(int(x.strip()) for x in retry_delays_str.split(","))
            except Exception:
                pass
        
        for attempt in range(len(retry_delays)):
            logger.info(f"Fetching MusicBrainz for: '{artist_name}'...")
            time.sleep(1.5 + (attempt * 2)) # standard pacing
            search_url = f"https://musicbrainz.org/ws/2/artist/?query=artist:\"{artist_name}\"&fmt=json"
            try:
                logger.debug(f"[MB Search] GET {search_url} | Headers: {headers}")
                res_obj = requests.get(search_url, headers=headers, timeout=10)
                logger.debug(f"[MB Search] Response {res_obj.status_code}: {res_obj.text[:200]}")
                if res_obj.status_code == 200:
                    res = res_obj.json()
                    if not res.get("artists"):
                        self.cache[key] = []
                        self._save_cache()
                        return []
                    mbid = res["artists"][0]["id"]
                    break
                elif res_obj.status_code == 503:
                    delay = retry_delays[attempt]
                    logger.warning(f"MusicBrainz API returned 503, waiting {delay}s before retrying ({attempt+1}/{len(retry_delays)})... | Body: {res_obj.text[:100]}")
                    time.sleep(delay)
                else:
                    logger.warning(f"MusicBrainz API returned {res_obj.status_code} | Body: {res_obj.text[:100]}")
                    return []
            except requests.exceptions.RequestException as e:
                delay = retry_delays[attempt]
                logger.warning(f"Network error on MB search, waiting {delay}s before retrying ({attempt+1}/{len(retry_delays)})... Error: {e}")
                time.sleep(delay)
                
        if not mbid:
            return []
            
        for attempt in range(len(retry_delays)):
            time.sleep(1.5 + (attempt * 2))
            detail_url = f"https://musicbrainz.org/ws/2/artist/{mbid}?inc=aliases+artist-rels&fmt=json"
            try:
                logger.debug(f"[MB Detail] GET {detail_url}")
                detail_obj = requests.get(detail_url, headers=headers, timeout=10)
                logger.debug(f"[MB Detail] Response {detail_obj.status_code}: {detail_obj.text[:200]}")
                if detail_obj.status_code == 200:
                    detail = detail_obj.json()
                    aliases = set()
                    for alias in detail.get("aliases", []):
                        if alias.get("name"):
                            aliases.add(alias["name"])
                            
                    for rel in detail.get("relations", []):
                        if rel.get("type") == "member of band" and rel.get("direction") == "forward":
                            band = rel.get("artist", {})
                            if band.get("name"):
                                aliases.add(band["name"])
                                
                    aliases_list = sorted(list(aliases))
                    self.cache[key] = aliases_list
                    self._save_cache()
                    logger.info(f"MB Fetch finished in {time.time()-t0:.2f}s (found {len(aliases_list)} aliases)")
                    return aliases_list
                elif detail_obj.status_code == 503:
                    delay = retry_delays[attempt]
                    logger.warning(f"MusicBrainz API returned 503 for details, waiting {delay}s before retrying ({attempt+1}/{len(retry_delays)})... | Body: {detail_obj.text[:100]}")
                    time.sleep(delay)
                else:
                    logger.warning(f"MusicBrainz API returned {detail_obj.status_code} for details | Body: {detail_obj.text[:100]}")
                    return []
            except requests.exceptions.RequestException as e:
                delay = retry_delays[attempt]
                logger.warning(f"Network error on MB details, waiting {delay}s before retrying ({attempt+1}/{len(retry_delays)})... Error: {e}")
                time.sleep(delay)
                
        return []


class MatchingEngine:
    """Point-based matcher: every signal earns 0-1 points, weighted and summed. No kill switches.

    Signals: title similarity, length-relative duration, artist, YouTube search rank,
    version agreement (live/studio/etc.), and source (song vs video).
    """

    def __init__(self, ytmusic: Optional[YTMusic] = None) -> None:
        auth_path = PROJECT_ROOT / "browser.json"
        if ytmusic is not None:
            self.yt = ytmusic
        elif auth_path.exists():
            self.yt = YTMusic(str(auth_path))
        else:
            self.yt = YTMusic()
            
        self.alias_manager = AliasManager(PROJECT_ROOT / "data" / "state" / "artist_aliases.json")

    # ------------------------------------------------------------------ signals
    @staticmethod
    def title_score(sp: ParsedTitle, yt: ParsedTitle) -> float:
        """Blend of token_set (tolerates extra words) and token_sort (keeps short titles honest)."""
        if not sp.core_norm or not yt.core_norm:
            return 0.0
        set_r = fuzz.token_set_ratio(sp.core_norm, yt.core_norm) / 100.0
        sort_r = fuzz.token_sort_ratio(sp.core_norm, yt.core_norm) / 100.0
        return 0.7 * set_r + 0.3 * sort_r

    @staticmethod
    def full_title_score(sp_full: str, yt_full: str) -> float:
        """Similarity of the raw, un-cleansed titles to ensure extra tags match."""
        sp_norm = normalize(sp_full)
        yt_norm = normalize(yt_full)
        if not sp_norm or not yt_norm: return 0.0
        return fuzz.token_sort_ratio(sp_norm, yt_norm) / 100.0

    @staticmethod
    def duration_score(diff_seconds: int, song_length_seconds: int) -> float:
        tolerance = max(5.0, 0.05 * song_length_seconds)
        return math.exp(-((diff_seconds / tolerance) ** 2))

    def is_valid_artist(self, sp_artists_raw: List[str], yt_names: List[str], artist_in_title: bool, aliases: List[str] = None) -> bool:
        if artist_in_title:
            return True
        
        sp_names = [normalize(a) for a in sp_artists_raw]
        for s in sp_names:
            for y in yt_names:
                if not s or not y:
                    continue
                # If YouTube artist contains bad tags (e.g. tribute, karaoke, cover) not in original artist, ignore
                if any(t in BAD_TAGS and t not in s.split() for t in y.split()):
                    continue
                if _contains_words(y, s) or _contains_words(s, y):
                    return True
                if fuzz.ratio(s, y) > 85:  # Allow minor typos, e.g. 'U2' vs 'U-2'
                    return True
                
        # Check aliases if provided
        if aliases:
            for alias in aliases:
                norm_alias = normalize(alias)
                if not norm_alias: continue
                for y in yt_names:
                    if any(t in BAD_TAGS and t not in norm_alias.split() for t in y.split()):
                        continue
                    if _contains_words(y, norm_alias) or _contains_words(norm_alias, y):
                        return True
                    if fuzz.ratio(norm_alias, y) > 85:
                        return True

        return False

    @staticmethod
    def rank_score(position: int) -> float:
        """position is 0-based within one search result list."""
        return max(0.0, 1.0 - position * 0.10)

    @staticmethod
    def version_score(sp_tags: Set[str], yt_tags: Set[str]) -> float:
        if (yt_tags & BAD_TAGS) - sp_tags:
            return 0.0
        score = 1.0
        if bool(sp_tags & LIVE_TAGS) != bool(yt_tags & LIVE_TAGS):
            score *= 0.6
        if (sp_tags & ALT_TAGS) != (yt_tags & ALT_TAGS):
            score *= 0.5
        return score

    # ------------------------------------------------------------------ scoring
    def score_result(
        self,
        track: SpotifyTrack,
        sp_parsed: ParsedTitle,
        sp_artists: List[str],
        item: Dict[str, Any],
        position: int,
        aliases: List[str] = None
    ) -> Optional[Candidate]:
        yt_dur = item.get("duration_seconds")
        video_id = item.get("videoId")
        if yt_dur is None or not video_id:
            return None

        is_video = item.get("resultType") == "video"
        raw_title = item.get("title", "") or ""
        yt_artist_raw = [a.get("name", "") for a in item.get("artists", []) or [] if a.get("name")]
        if item.get("author"):
            yt_artist_raw.append(item["author"])
        yt_artists = [normalize(a) for a in yt_artist_raw]

        yt_parsed, artist_removed = parse_title(raw_title, sp_artists if is_video else None)
        # Karaoke/cover channels often only say so in the artist name, or we might have stripped it
        # accidentally from the title if it was "(Artist cover)". Ensure we catch BAD_TAGS everywhere.
        for name in yt_artists:
            yt_parsed.tags.update(t for t in name.split() if t in BAD_TAGS)
        for token in normalize(raw_title).split():
            if token in BAD_TAGS:
                yt_parsed.tags.add(token)

        sp_tokens = set(normalize(track.title).split())
        has_bad_tags = any(t in BAD_TAGS and t not in sp_tokens for t in yt_parsed.tags)

        artist_in_title = is_video and not has_bad_tags and (
            artist_removed or any(_contains_words(normalize(raw_title), a) for a in sp_artists)
        )

        # HARD RULE: The artist must match (either exact, in title, or alias).
        # It's a binary gate. If it's a cover band, discard entirely.
        if not self.is_valid_artist(track.artists, yt_artists, artist_in_title, aliases):
            logger.debug(f"Discarding '{raw_title}' - Artist mismatch")
            return None

        diff = abs(int(yt_dur) - track.duration_seconds)
        
        # Raw Video Type (One-Hot)
        vtype = item.get("videoType", "")
        is_atv = 1.0 if vtype == "MUSIC_VIDEO_TYPE_ATV" else 0.0
        is_omv = 1.0 if vtype == "MUSIC_VIDEO_TYPE_OMV" else 0.0
        is_ugc = 1.0 if vtype == "MUSIC_VIDEO_TYPE_UGC" else 0.0
        
        # Raw Album Signals
        sp_album = getattr(track, "album", None)
        has_sp_album = 1.0 if (sp_album and sp_album.strip()) else 0.0
        yt_album = item.get("album", {}).get("name") if isinstance(item.get("album"), dict) else item.get("album")
        has_yt_album = 1.0 if (yt_album and str(yt_album).strip()) else 0.0
        
        album_sim = (fuzz.token_set_ratio(normalize(sp_album), normalize(str(yt_album))) / 100.0) if (has_sp_album and has_yt_album) else 0.0
        album_in_title = 1.0 if (has_sp_album and normalize(sp_album) in normalize(raw_title)) else 0.0

        is_live_mismatch = 1.0 if ("live" not in sp_parsed.tags and "live" in yt_parsed.tags) else 0.0

        breakdown = {
            "title": self.title_score(sp_parsed, yt_parsed),
            "full_title": self.full_title_score(track.title, raw_title),
            "duration": self.duration_score(diff, track.duration_seconds),
            "rank": self.rank_score(position),
            "version": self.version_score(sp_parsed.tags, yt_parsed.tags),
            "is_live_mismatch": is_live_mismatch,
            "is_video": 1.0 if is_video else 0.0,
            "is_atv": is_atv,
            "is_omv": is_omv,
            "is_ugc": is_ugc,
            "has_spotify_album": has_sp_album,
            "has_yt_album": has_yt_album,
            "album_similarity": album_sim,
            "album_in_title": album_in_title,
        }
        
        # Hard gate: The title must at least somewhat resemble the search query
        if breakdown["title"] < 0.25 and breakdown["full_title"] < 0.25:
            return None
            
        features = list(breakdown.values())
        n_expected = getattr(ML_MODEL, "n_features_in_", 0) if ML_MODEL is not None else 0
        if n_expected == len(features):
            total = float(ML_MODEL.predict_proba([features])[0][1])
        elif n_expected == 6:
            # Fallback for old 6-feature model before retraining
            old_features = [breakdown["title"], breakdown["full_title"], breakdown["duration"], breakdown["rank"], breakdown["version"], 0.0 if is_video else 1.0]
            total = float(ML_MODEL.predict_proba([old_features])[0][1])
        else:
            total = sum(WEIGHTS.get(k, 0.0) * v for k, v in breakdown.items())

        return Candidate(
            track=YouTubeTrack(
                video_id=video_id,
                title=raw_title,
                artist=", ".join(yt_artist_raw),
                duration_seconds=int(yt_dur),
                match_type="video" if is_video else "song",
            ),
            score=total,
            duration_diff=diff,
            breakdown={k: round(v, 4) for k, v in breakdown.items()},
        )

    # ------------------------------------------------------------------ search
    def _search(self, query: str, result_filter: str) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
            try:
                pacing = random.uniform(*SEARCH_PACING_RANGE_SECONDS)
                time.sleep(pacing)
                logger.info(f"Searching YT ({result_filter}): '{query}'...")
                t0 = time.time()
                res = self.yt.search(query, filter=result_filter, limit=MAX_RESULTS_PER_SEARCH)
                logger.info(f"YT Search finished in {time.time()-t0:.2f}s (found {len(res) if res else 0} results)")
                return (res or [])[:MAX_RESULTS_PER_SEARCH]
            except Exception as e:  # network / throttling / parsing
                last_error = e
                if attempt < len(RETRY_DELAYS_SECONDS):
                    backoff = RETRY_DELAYS_SECONDS[attempt]
                    logger.warning(f"Search failed: {last_error}. Retrying in {backoff}s...")
                    time.sleep(backoff)
        raise SearchError(f"Search failed after {len(RETRY_DELAYS_SECONDS)} retries for '{query}' ({result_filter}): {last_error}")

    @staticmethod
    def build_queries(track: SpotifyTrack, sp_parsed: ParsedTitle) -> List[str]:
        primary = primary_artist_display(track)
        all_artists = " ".join(p.strip() for a in track.artists for p in ARTIST_SPLIT.split(a) if p.strip())
        core = sp_parsed.core_display
        with_tags = f"{core} {' '.join(sorted(sp_parsed.tags))}".strip()

        queries = [
            f"{primary} {core}".strip(),
            f"{all_artists} {core}".strip(),
            f"{primary} {with_tags}".strip(),
        ]
        seen: Set[str] = set()
        return [q for q in queries if not (q.lower() in seen or seen.add(q.lower()))]

    # ------------------------------------------------------------------ match
    def match(self, track: SpotifyTrack, mock_results: Optional[List[Dict[str, Any]]] = None) -> TrackMatch:
        """Score every candidate from songs then videos, keep the best. Raises SearchError on persistent failure."""
        sp_parsed, _ = parse_title(track.title)
        sp_artists = split_artists(track.artists)
        aliases: List[str] = []
        if track.artists:
            for art in track.artists:
                aliases.extend(self.alias_manager.fetch_aliases(art))
        best: Optional[Candidate] = None

        if mock_results is not None:
            # Offline simulation mode using pre-fetched dataset
            song_pos = 0
            video_pos = 0
            for item in mock_results:
                is_vid = item.get("resultType", "video").upper() != "SONG"
                pos = video_pos if is_vid else song_pos
                if is_vid:
                    video_pos += 1
                else:
                    song_pos += 1
                cand = self.score_result(track, sp_parsed, sp_artists, item, pos, aliases=aliases)
                if cand and (best is None or cand.score > best.score):
                    best = cand
        else:
            # Live search mode
            for query in self.build_queries(track, sp_parsed):
                try:
                    songs = self._search(query, "songs")
                    for pos, item in enumerate(songs):
                        if item.get("resultType") == "song":
                            cand = self.score_result(track, sp_parsed, sp_artists, item, pos, aliases=aliases)
                            if cand and (best is None or cand.score > best.score):
                                best = cand
                except Exception as e:
                    logger.warning(f"Failed to search songs for {query}: {e}")
                    
                try:
                    videos = self._search(query, "videos")
                    for pos, item in enumerate(videos):
                        if item.get("resultType") == "video":
                            cand = self.score_result(track, sp_parsed, sp_artists, item, pos, aliases=aliases)
                            if cand and (best is None or cand.score > best.score):
                                best = cand
                except Exception as e:
                    logger.warning(f"Failed to search videos for {query}: {e}")
                        
                if best and best.score >= EARLY_STOP_THRESHOLD:
                    break

        if best and best.score >= MIN_CONFIDENCE_THRESHOLD:
            return TrackMatch(
                spotify_track=track,
                youtube_track=best.track,
                confidence_score=best.score,
                duration_diff_seconds=best.duration_diff,
                breakdown=best.breakdown,
            )
        return TrackMatch(
            spotify_track=track,
            youtube_track=None,
            confidence_score=best.score if best else 0.0,
            duration_diff_seconds=None,
            breakdown=best.breakdown if best else None,
            best_rejected=best.track if best else None,
        )
