# Audio Auto-Tagging and Similarity Search

Upload an audio file and get sound tags with confidence scores. Each file's audio embedding is stored in a vector index, so you can find similar-sounding files or search your library by tag. The system runs as containerised services with Docker Compose.

![Tagging a track](screenshots/Screenshot%202024-12-17%20at%2000.31.28.png)

*Team project for the Audio Processing and Indexing course, MSc Media Technology, Leiden University (December 2024). Built with Shreyansh Sharma, H.P. Pranaav and Trent Eriksen.*

## The problem

Tagging audio libraries by hand is slow, and it is the bottleneck for anyone who needs to search a large collection of sound effects, stems or recordings. The goal was to tag new files automatically, with no training on the user's own data, and to make the collection searchable by label and by sound.

## Architecture

```
Streamlit UI (8501) ──upload──► Tagging API, FastAPI (8000) ──► ConvNeXt-Tiny (AudioSet)
      │                                   │                         ├─ 527-class probabilities
      │                                   │                         └─ 768-d scene embedding
      │                                   ▼
      └──label search / k-NN──► OpenSearch (9200), k-NN index (HNSW)
```

- **Tagging** (`src/tagging/tagging.py`): audio is resampled to 32 kHz and split into 10-second windows. Each window goes through [ConvNeXt-Tiny trained on AudioSet](https://huggingface.co/topel/ConvNeXt-Tiny-AT). Labels above 0.25 probability are kept, and the scores are averaged per file.
- **Indexing:** each clip's 768-dimensional embedding, labels and probabilities are stored in OpenSearch, using a `knn_vector` field with an HNSW index (nmslib, L2 distance).
- **UI** (`src/streamlit-ui/streamlit_app.py`): an **Index** tab to upload, play, view tags as a table and chart, export them as CSV, and find acoustically similar files with k-NN search. A **Search** tab queries the library by tag.
- **Deployment:** each service has its own Dockerfile, and `docker-compose.yml` starts all three.

## Results

- Tags a new file with no training data. For example, a Moog synthesiser track returned *Music* (0.63), *Musical instrument* (0.50), *Keyboard* (0.42), *Piano* (0.30) and *Synthesizer* (0.27). More examples are in [`screenshots/`](screenshots).
- On our test tracks, the tags matched the obvious musical content well: drums, guitar and synth, and effects such as reverb. This was checked by listening, not measured against a labelled test set.

## How to run

Requires Docker and Docker Compose.

```bash
cd src
docker-compose up
```

After a few minutes the UI is available at http://localhost:8501.

## Known limitations and next steps

- Only the first 10 seconds of each file are analysed. The next step is to tag the whole file with overlapping windows.
- Files shorter than 10 seconds cause an error.
- The OpenSearch credentials are the local development defaults in the compose file. Move them to a `.env` file before deploying anywhere.
- `src/indexing/audio_indexing.py` is an unfinished experiment with an alternative embedding model, and it isn't used by the running system.
- Next: evaluate tag accuracy against a labelled test set, for example AudioSet's evaluation split.
