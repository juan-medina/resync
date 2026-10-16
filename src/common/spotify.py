# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from typing import Optional, List
from spotify_scraper import SpotifyClient
from tqdm import tqdm
from src.common.models import SpotifyPlaylist, SpotifyTrack

# Setup progress tracking on SpotifyClient pagination
_CURRENT_PBAR = None
_orig_fetch_union = SpotifyClient._fetch_union


def _tracked_fetch_union(self, *args, **kwargs):
    res = _orig_fetch_union(self, *args, **kwargs)
    global _CURRENT_PBAR
    if _CURRENT_PBAR is not None:
        _CURRENT_PBAR.update(100)
    return res


SpotifyClient._fetch_union = _tracked_fetch_union


import logging
logger = logging.getLogger("resync")


def get_spotify_playlist(
    playlist_id_or_url: str,
    max_tracks: Optional[int] = None,
    show_progress: bool = True,
) -> SpotifyPlaylist:
    """Fetch playlist metadata and tracks from Spotify with progress reporting.

    Uses anonymous web tokens directly, requiring no Spotify Developer
    credentials or Premium account.

    Args:
        playlist_id_or_url: Spotify playlist URL, URI, or ID.
        max_tracks: Maximum number of tracks to fetch. If None, fetches all.
        show_progress: Whether to show a tqdm progress bar during fetching.

    Returns:
        SpotifyPlaylist domain object containing parsed tracks.
    """
    clean_id = (playlist_id_or_url or "").strip()
    if not clean_id:
        msg = "Spotify playlist ID or URL cannot be empty."
        logger.error(msg)
        raise ValueError(msg)

    global _CURRENT_PBAR
    client = SpotifyClient()

    try:
        if show_progress:
            with tqdm(desc="Fetching Spotify Playlist", unit=" tracks") as pbar:
                _CURRENT_PBAR = pbar
                raw_playlist = client.get_playlist(clean_id, max_tracks=max_tracks)
                pbar.total = len(raw_playlist.tracks)
                pbar.n = len(raw_playlist.tracks)
                pbar.refresh()
                _CURRENT_PBAR = None
        else:
            raw_playlist = client.get_playlist(clean_id, max_tracks=max_tracks)
    except Exception as e:
        logger.error(f"Error fetching Spotify playlist '{clean_id}': {e}", exc_info=True)
        raise

    parsed_tracks: List[SpotifyTrack] = []
    for item in raw_playlist.tracks:
        if item.track is None:
            continue
        t = item.track
        artists = [artist.name for artist in t.artists] if t.artists else []
        album_name = getattr(t.album, "name", None) if getattr(t, "album", None) else None
        parsed_tracks.append(
            SpotifyTrack(
                id=t.id,
                title=t.name,
                artists=artists,
                duration_ms=t.duration_ms,
                album=album_name,
            )
        )

    return SpotifyPlaylist(
        id=raw_playlist.id,
        name=raw_playlist.name,
        total_tracks=raw_playlist.total_tracks,
        tracks=parsed_tracks,
    )
