# Project Context: Spotify to YouTube Music Playlist Sync

## Overview
This project is a Python tool designed to synchronize a third-party Spotify playlist to a personal YouTube Music playlist on demand. It employs a Machine Learning classifier to evaluate search candidates and select the correct YouTube Music track version (Studio, Live, Remaster, etc.) without relying on brittle hardcoded rules.

## Technical Stack
* **Language:** Python 3.10+
* **Spotify API:** `spotipy` (OAuth and playlist retrieval)
* **YouTube Music API:** `ytmusicapi` (Playlist management and search)
* **Machine Learning:** `scikit-learn` (Random Forest Classifier for candidate ranking)
* **State Management:** Local JSON files for diff tracking, state caching, and artist alias resolution.

## Core Directives & Rules
1. **Matching Accuracy is Priority #1:** YouTube search results contain varied metadata quality. The system evaluates candidates from both "Songs" and "Videos" tabs concurrently, allowing the ML model to score candidates based on title, duration, tags, rank, and album context.
2. **Probability-Driven Scoring:** Candidates must be ranked using exact unrounded model probabilities to preserve fine-grained probabilistic separation and prevent artificial ties.
3. **Deduplication:** Ensure the final playlist reflects the unique tracks intended by the user, resolving redundant matches to prevent consecutive duplicate entries.
4. **State Diffing:** Calculate additions and deletions using `data/state/sync_state.json`. Only apply required diffs to YouTube Music rather than recreating playlists. Preserve dated backup snapshots before full rebuilds.
5. **Separation of Concerns:** Maintain a clean separation between the ML Training Pipeline (candidate dumping, labeling, feature extraction, model fitting) and the Production Sync Tool (query construction, inference, playlist diffing).
6. **High-Level Abstraction:** Focus on architecture and data flow. Avoid introducing brittle, low-level binary gates (such as arbitrary duration cutoffs) that bypass model inference.

---

## Agent Roles

### Agent: `API_Specialist`
* **Role:** Manages all external communication with Spotify and YouTube Music.
* **Responsibilities:**
  * Manage authentication workflows (`spotipy` OAuth, `ytmusicapi` session headers/cookies).
  * Provide clean, throttled wrappers for fetching playlist contents, search queries, and modifying playlists.
  * Handle pagination, rate limits, and transient API errors.

### Agent: `Matching_Engine`
* **Role:** Translates a Spotify track into the single best matching YouTube Music track ID.
* **Responsibilities:**
  * Construct clean, noise-reduced search queries.
  * Fetch candidate pools spanning both Songs and Videos.
  * Leverage local alias cache (`data/state/artist_aliases.json`) to resolve artist name variations without network overhead.
  * Extract standardized comparative features and delegate final candidate scoring to the trained ML model.

### Agent: `Sync_Manager`
* **Role:** Manages state persistence, playlist diffing, and execution safety.
* **Responsibilities:**
  * Manage `data/state/sync_state.json` and create dated backups prior to major operations.
  * Compare current Spotify playlist items against local state and target YouTube playlist state.
  * Compute minimal diffs (tracks to add, tracks to remove) and deduplicate consecutive matches.
  * Orchestrate updates through `API_Specialist` and update the local sync state upon completion.
