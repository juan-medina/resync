# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_default_auth_path() -> Path:
    if (PROJECT_ROOT / "browser.json").exists():
        return PROJECT_ROOT / "browser.json"
    if (PROJECT_ROOT / "oauth.json").exists():
        return PROJECT_ROOT / "oauth.json"
    return PROJECT_ROOT / "browser.json"


class YouTubeMusicClient:
    """Handles authenticated playlist management on YouTube / YouTube Music.
    
    Uses official YouTube Data API v3 when authenticated via Google OAuth (oauth.json),
    and falls back to ytmusicapi when using browser cookies (browser.json).
    """

    def __init__(self, auth_path: Optional[Path] = None) -> None:
        self.auth_path = auth_path or get_default_auth_path()
        if not self.auth_path.exists():
            raise FileNotFoundError(
                f"Authentication file '{self.auth_path}' not found.\n"
                f"Please run '.\\scripts\\setup_auth.bat' to generate 'browser.json'."
            )

        self.is_oauth = "oauth" in self.auth_path.stem or self._check_is_oauth()
        if not self.is_oauth:
            from ytmusicapi import YTMusic
            self.yt = YTMusic(str(self.auth_path))
        else:
            self._ensure_valid_token()

    def _check_is_oauth(self) -> bool:
        try:
            with open(self.auth_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return "access_token" in data or "refresh_token" in data
        except Exception:
            return False

    def _ensure_valid_token(self) -> str:
        """Verify token expiration and refresh automatically if needed."""
        with open(self.auth_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expires_at = data.get("expires_at", 0)
        # Refresh if token expires within 2 minutes
        if time.time() > expires_at - 120 and "refresh_token" in data:
            refresh_resp = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": data.get("client_id"),
                    "client_secret": data.get("client_secret"),
                    "refresh_token": data.get("refresh_token"),
                    "grant_type": "refresh_token",
                },
            )
            if refresh_resp.status_code == 200:
                fresh = refresh_resp.json()
                data["access_token"] = fresh["access_token"]
                data["expires_in"] = fresh.get("expires_in", 3600)
                data["expires_at"] = int(time.time()) + data["expires_in"]
                with open(self.auth_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                print(f"Warning: Failed to refresh token: {refresh_resp.text}")

        self.access_token = data.get("access_token", "")
        return self.access_token

    def _get_headers(self) -> Dict[str, str]:
        token = self._ensure_valid_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def check_auth(self, target_playlist_id: Optional[str] = None) -> None:
        """Validate authentication before starting sync.
        
        Raises PermissionError or RuntimeError with human-friendly instructions if
        session is invalid, expired, or does not own the target playlist.
        """
        if self.is_oauth:
            try:
                self._ensure_valid_token()
            except Exception as e:
                raise PermissionError(
                    f"Google OAuth token is invalid or expired: {e}\n"
                    f"Please run '.\\scripts\\setup_auth.bat' to re-authenticate."
                )
            return

        # browser.json (ytmusicapi) check
        if target_playlist_id:
            try:
                data = self.yt.get_playlist(target_playlist_id, limit=1)
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to YouTube Music for playlist '{target_playlist_id}': {e}\n"
                    f"Please verify your internet connection and playlist ID."
                )

            if not data.get("owned", False):
                title = data.get("title", target_playlist_id)
                raise PermissionError(
                    f"Authentication expired or invalid: YouTube Music treated your session as a guest viewer.\n"
                    f"You do not have write/edit permissions for playlist '{title}' ({target_playlist_id}).\n"
                    f"Please run '.\\scripts\\setup_auth.bat' to refresh your browser.json authentication."
                )
        else:
            try:
                self.yt.get_liked_songs(limit=1)
            except Exception:
                raise PermissionError(
                    f"Authentication expired or invalid: YouTube Music session is not signed in.\n"
                    f"Please run '.\\scripts\\setup_auth.bat' to refresh your browser.json authentication."
                )


    def create_playlist(
        self,
        title: str,
        description: str = "Synced with resync",
        privacy_status: str = "PRIVATE",
        video_ids: Optional[List[str]] = None,
    ) -> str:
        """Create a new playlist and return its playlist ID."""
        if not self.is_oauth:
            playlist_id: str = self.yt.create_playlist(
                title=title,
                description=description,
                privacy_status=privacy_status,
                video_ids=video_ids,
            )
            return playlist_id

        url = "https://www.googleapis.com/youtube/v3/playlists?part=snippet,status"
        payload = {
            "snippet": {
                "title": title,
                "description": description,
            },
            "status": {
                "privacyStatus": privacy_status.lower(),
            },
        }

        resp = requests.post(url, headers=self._get_headers(), json=payload)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create playlist ({resp.status_code}): {resp.text}")

        playlist_id = resp.json()["id"]

        if video_ids:
            self.add_tracks(playlist_id, video_ids)

        return playlist_id

    def add_tracks(self, playlist_id: str, video_ids: List[str]) -> None:
        """Add a list of video IDs to an existing playlist."""
        if not video_ids:
            return

        if not self.is_oauth:
            batch_size = 50
            batches = [video_ids[i:i + batch_size] for i in range(0, len(video_ids), batch_size)]
            for batch in tqdm(batches, desc="Adding tracks to YT Music (batches of 50)", unit="batch"):
                self.yt.add_playlist_items(playlist_id, batch, duplicates=True)
            return

        url = "https://www.googleapis.com/youtube/v3/playlistItems?part=snippet"
        headers = self._get_headers()

        quota_reached = False
        with tqdm(video_ids, desc="Adding tracks to YouTube Music", unit="track") as pbar:
            for vid in pbar:
                payload = {
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": vid,
                        },
                    }
                }
                resp = requests.post(url, headers=headers, json=payload)

                if resp.status_code in (200, 201):
                    continue

                if resp.status_code == 403:
                    err_json = resp.json()
                    errors = err_json.get("error", {}).get("errors", [])
                    reasons = [e.get("reason") for e in errors]
                    if "quotaExceeded" in reasons:
                        quota_reached = True
                        print(
                            f"\n[Warning] YouTube Data API daily quota limit reached (10,000 units).\n"
                            f"Added remaining items before limit. The rest can be added on next run."
                        )
                        break

                # Ignore individual video add failures (e.g. video unavailable/duplicate)
                pbar.write(f"Notice: Could not add video {vid} ({resp.status_code}): {resp.text[:100]}")

    def remove_tracks(self, playlist_id: str, items: List[Dict[str, Any]]) -> None:
        """Remove a list of tracks from an existing playlist."""
        if not items:
            return

        if not self.is_oauth:
            # ytmusicapi requires a list of dicts with 'videoId' and 'setVideoId'
            missing_set_ids = [item for item in items if not item.get("setVideoId")]
            if missing_set_ids:
                raise PermissionError(
                    f"Cannot remove {len(missing_set_ids)} tracks: missing playlist ownership tokens ('setVideoId').\n"
                    f"YouTube Music treated your session as a guest viewer because your browser.json authentication is expired.\n"
                    f"Please run '.\\scripts\\setup_auth.bat' to refresh your login."
                )

            batch_size = 50
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            for batch in tqdm(batches, desc="Removing tracks from YT Music (batches of 50)", unit="batch"):
                self.yt.remove_playlist_items(playlist_id, batch)
            return

        # OAuth via Google Data API
        url = "https://www.googleapis.com/youtube/v3/playlistItems"
        headers = self._get_headers()
        
        with tqdm(items, desc="Removing tracks from YT Music", unit="track") as pbar:
            for item in pbar:
                item_id = item.get("playlistItemId")
                if not item_id:
                    continue
                
                resp = requests.delete(url, headers=headers, params={"id": item_id})
                if resp.status_code not in (200, 204):
                    pbar.write(f"Notice: Could not remove item {item_id} ({resp.status_code}): {resp.text[:100]}")

    def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Fetch all current tracks from a playlist."""
        if not self.is_oauth:
            playlist_data = self.yt.get_playlist(playlist_id, limit=None)
            return playlist_data.get("tracks", [])

        tracks: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        base_url = "https://www.googleapis.com/youtube/v3/playlistItems"

        while True:
            params: Dict[str, Any] = {
                "part": "snippet,contentDetails",
                "playlistId": playlist_id,
                "maxResults": 50,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = requests.get(base_url, headers=self._get_headers(), params=params)
            if resp.status_code != 200:
                break

            data = resp.json()
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                vid = snippet.get("resourceId", {}).get("videoId")
                if vid:
                    tracks.append({
                        "videoId": vid,
                        "title": snippet.get("title", ""),
                        "playlistItemId": item.get("id"),
                    })

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return tracks
