# ImageDB — Text-based Image Retrieval

Convert folder(s) and/or files(s) of photos into a searchable image database (text queries → visually similar images).  
This project embeds images and text using a CLIP-based Sentence-Transformer and stores image vectors in FAISS for fast nearest-neighbor search.

**Files**
- `ImageDB.py` — index and query functionality (embeds images, writes FAISS index and metadata).
- `imagedbCLI.py` — small CLI wrapper for indexing and querying.

## Requirements

See `requirements.txt`. Recommended Python >= 3.9.

## Install

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

2. (Optional GPU) Install `torch` and `faiss-gpu` following official install instructions for your CUDA version.

## Usage

### Index images
Put all images inside a folder, e.g. `images/`. Then run:
```bash
python imagedbCLI.py index images/
```
This creates:
- `img_index.faiss` — FAISS index
- `image_paths.npy` — array of file paths

### Query the database
Search with text, top-k results:
```bash
python imagedbCLI.py query "dog" -k 8
```
Example output:
```
Image: images/dog1.jpg,    Score: 0.8237
...
```

## How it works (brief)
- Uses `sentence-transformers` with `clip-ViT-B-32` to create embeddings for images and text (same embedding space).
- Stores normalized image vectors in FAISS (IndexFlatIP) for cosine similarity search.
- CLI calls `ImageDB.index(...)` and `ImageDB.query_image(...)`.


## Troubleshooting
- **No results / low scores**: Recheck that embeddings were created successfully and that `img_index.faiss` and `image_paths.npy` exist in the same folder where you run queries.
- **Model download errors**: The first run downloads model weights; ensure internet access and enough disk space.


## Create GitHub repo (local -> remote)
```bash
git init
git add .
git commit -m "Initial commit: ImageDB + CLI"
# create repo on github.com and then:
git remote add origin git@github.com:yourusername/your-repo-name.git
git branch -M main
git push -u origin main
```
