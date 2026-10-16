# Resync

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)

Resync is a Python tool that synchronizes a third-party Spotify playlist to a personal YouTube Music playlist on demand. It uses a trained Machine Learning model (Random Forest) to evaluate search candidates from YouTube Music (both songs and videos) and select the true equivalent track (Studio, Live, Remaster, Acoustic) while rejecting unwanted covers, amateur remixes, and algorithm noise.

For full details on feature extraction, heuristics, and model behavior, see the [Model Card](MODEL_CARD.md).

## Key Features
* **ML Candidate Selection**: Evaluates title similarity, duration proximity, version tags, search rank, and album metadata to pick the best match.
* **Diff-Based Sync**: Tracks synced tracks in `data/state/sync_state.json` to calculate minimal additions and deletions without rebuilding the target playlist.
* **Deduplication**: Resolves duplicate canonical entries to prevent consecutive identical tracks.
* **Artist Alias Resolution**: Uses a local MusicBrainz alias cache to identify tracks released under alternate artist names or collaborations.
* **Dry Run Mode**: Test queries and review match decisions before modifying your live YouTube playlist.

## Prerequisites
* Python 3.10+
* Spotify Developer Application credentials (`SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`)
* YouTube Music authentication session (`oauth.json` or browser request headers)
* *(Optional)* Gemini API Key (`GEMINI_API_KEY`) — only required if retraining the model from scratch.

## Setup
1. Clone the repository and configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Fill in your Spotify API credentials, playlist IDs, and sync options in `.env`.

2. Install dependencies:
   ```bash
   make setup
   ```

3. Authenticate with Spotify and YouTube Music:
   ```bash
   make auth
   ```

## Usage

### Production Sync
To run the sync using the pre-trained model:
```bash
make sync
```
To preview changes without altering the YouTube playlist, set `DRY_RUN=true` in `.env`.

### ML Training Pipeline (Optional)
If you wish to retrain the matcher model from scratch:
1. **Dump Search Dataset**: Retrieve candidate pools from YouTube Music for training tracks:
   ```bash
   make dump
   ```
2. **Auto-Label Dataset**: Use LLM evaluation to label correct matches:
   ```bash
   make label
   ```
3. **Train Matcher**: Fit the Random Forest classifier and save the artifact to `models/matcher_model.pkl`:
   ```bash
   make train
   ```

## Documentation
* [Model Card](MODEL_CARD.md) — Model architecture, 14-feature specifications, and retrieval logic.
* [Agent Guidelines](AGENTS.md) — Architecture overview, core directives, and agent role responsibilities.

## License
[MIT License](LICENSE) © 2026 Juan Medina
