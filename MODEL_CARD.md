# Model Card: Resync Matcher Model

## Model Details
* **Model Type:** Random Forest Classifier (`scikit-learn`)
* **Task:** Pointwise candidate ranking. Given a target Spotify track and a candidate pool from YouTube Music, predict the probability that a candidate is the true equivalent.
* **Hyperparameters:** `n_estimators=100`, `max_depth=8`, `min_samples_leaf=10`, `random_state=42`. Constrained tree depth ensures robust generalization across imbalanced candidate sets (1 positive per 39 negative candidates).
* **Output:** Continuous probability between `0.0` and `1.0`. The candidate with the highest score above `0.25` is selected. Raw unrounded floating-point probabilities are compared to prevent artificial ties.

## Intended Use
This model automates synchronization between Spotify playlists and personal YouTube Music playlists. It identifies the correct track version (Studio, Live, Remaster, Acoustic) while rejecting unwanted covers, amateur remixes, and algorithm hallucinations.

## Feature Set (14 Features)
The model holistically evaluates 14 comparative features extracted from Spotify and YouTube Music metadata:

### 1. Title & Text Similarity
* **`title`**: Fuzzy token set similarity between cleaned track titles (stripping standard metadata noise like "(Remastered)").
* **`full_title`**: Fuzzy similarity between raw, unstripped title strings.

### 2. Temporal & Search Position
* **`duration`**: Normalized proximity score based on the absolute duration difference in seconds.
* **`rank`**: Normalized position within the search result category (`0.0` to `1.0`, where `1.0` represents top rank).

### 3. Version & Acoustic Alignment
* **`version`**: Semantic alignment score matching version modifiers (e.g., `Live`, `Radio Edit`, `Acoustic`, `Instrumental`).
* **`is_live_mismatch`**: Indicator flag set when one track is marked Live and the other is marked Studio.

### 4. Content Classification
* **`is_atv`**: Flag indicating an official Auto-Generated Topic Video (album audio release).
* **`is_omv`**: Flag indicating an Official Music Video.
* **`is_ugc`**: Flag indicating User-Generated Content or unofficial channel upload.
* **`is_video`**: Flag indicating whether the candidate originated from the YouTube "Videos" tab rather than "Songs".

### 5. Album Context
* **`has_spotify_album`**: Boolean indicator if Spotify provided album metadata.
* **`has_yt_album`**: Boolean indicator if YouTube Music provided album metadata.
* **`album_similarity`**: Fuzzy string similarity between Spotify and YouTube album names.
* **`album_in_title`**: Flag set when the Spotify album name appears inside the YouTube title.

## Candidate Retrieval & Pre-Filtering
Before model inference, broad candidate pools are retrieved and pre-filtered:
* **Dual Pool Retrieval:** Fetches top results from both "Songs" (canonical audio) and "Videos" tabs simultaneously.
* **Artist Verification:** Candidate channel or artist credits must match the Spotify artist or a verified alias from the local MusicBrainz cache (`artist_aliases.json`).
* **Negative Tag Filtering:** Rejects unwanted covers, karaoke tracks, and fan tributes unless explicitly present in the Spotify query.
* **Relevance Floor:** Discards candidates with title similarity below `0.25` before inference.

## Training Data & Methodology
* **Dataset:** Generated from thousands of unique Spotify tracks paired with candidate pools fetched from YouTube Music search results.
* **Labeling:** Ground-truth labels were generated using an LLM evaluator cross-referencing artists, duration, release versions, and channel legitimacy.
* **Scoring:** Pointwise ranking ranks candidates per song by predicted probability $P(match = 1)$.

## Known Trade-offs
* **Album Masters vs Short Edits:** When an official studio album track ranks #1 with full album metadata, it may score slightly higher than a radio edit sitting lower in search results, prioritizing official studio master quality over early fade-outs.
* **Catalog Fallback:** When a track lacks an official audio streaming release (`is_atv`), the model naturally falls back to high-confidence official music videos (`is_omv`).
