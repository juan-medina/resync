# Copyright (c) 2026 Juan Medina
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import json
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_FILE = PROJECT_ROOT / "data" / "state" / "sync_state.json"


class StateManager:
    """Manages local state cache for Spotify track -> YouTube Music video mappings."""

    def __init__(self, filepath: Optional[Path] = None) -> None:
        self.filepath = filepath or STATE_FILE
        self._state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.filepath.exists():
            return {}
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self) -> None:
        """Persist cache to disk."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)

    def get(self, spotify_id: str) -> Optional[Dict[str, Any]]:
        return self._state.get(spotify_id)

    def set(self, spotify_id: str, match_data: Dict[str, Any]) -> None:
        self._state[spotify_id] = match_data

    def count(self) -> int:
        return len(self._state)
