# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SpotifyTrack:
    """Represents a Spotify track's core metadata for synchronization."""
    id: str
    title: str
    artists: List[str]
    duration_ms: int
    album: Optional[str] = None

    @property
    def artist_display(self) -> str:
        return ", ".join(self.artists) if self.artists else "Unknown Artist"

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else ""

    @property
    def duration_seconds(self) -> int:
        return round(self.duration_ms / 1000)


@dataclass(frozen=True)
class SpotifyPlaylist:
    """Represents a fetched Spotify playlist."""
    id: str
    name: str
    total_tracks: int
    tracks: List[SpotifyTrack]


@dataclass(frozen=True)
class YouTubeTrack:
    """Represents a matched YouTube Music song or video."""
    video_id: str
    title: str
    artist: str
    duration_seconds: int
    match_type: str = "song"  # "song" or "video"


@dataclass(frozen=True)
class TrackMatch:
    """Represents a validated match between Spotify and YouTube Music."""
    spotify_track: SpotifyTrack
    youtube_track: Optional[YouTubeTrack]
    confidence_score: float
    duration_diff_seconds: Optional[int]
    breakdown: Optional[Dict[str, float]] = None
    best_rejected: Optional[YouTubeTrack] = None

    @property
    def is_matched(self) -> bool:
        return self.youtube_track is not None
