import faiss
import torch
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PIL import Image

MODEL_NAME = "clip-ViT-B-32"
INDEX_PATH = "img_index.faiss"
META_PATH = "image_paths.npy"

model = None

class ImageDB():
  def index(self, *args):
    """
    Index images into the database.
    Args:
      *args: Variable length argument list containing paths of images and/or path to folder containing images that will all get indexed.
    """
    image_paths = []
    vectors = []

    self._initialise_model()

    for path in args:
      if os.path.isdir(path):
        for img_file in os.listdir(path):
          img_path = os.path.join(path, img_file)
          if os.path.isfile(img_path):
            self._get_image_embedding(img_path, image_paths, vectors)
      elif os.path.isfile(path):
        self._get_image_embedding(img_path, image_paths, vectors)
      else:
        print(f"Path {path} is not valid.")

    if not vectors:
        raise RuntimeError("No images indexed.")

    vectors = np.vstack([v.reshape(1,-1) for v in vectors]).astype('float32')
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # use Inner Product on normalized vectors for cosine
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, np.array(image_paths))

    print(f"Indexed {len(image_paths)} images.")

  def query_image(self, query, k=5):
    """
    Query the database for images according to the text description.
    Args:
      query (str): Text description to query the database.
      k (int): Number of top results to return. Default is 5.
    """
    self._initialise_model()

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
      raise RuntimeError("Index or metadata file not found. Please index images first.")

    index = faiss.read_index(INDEX_PATH)
    image_paths = np.load(META_PATH, allow_pickle=True)

    query_emb = model.encode(query, convert_to_tensor=False, show_progress_bar=False)
    query_emb = query_emb.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_emb)

    distances, indices = index.search(query_emb, k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
      results.append((image_paths[idx], score))
    return results
    
  def _get_image_embedding(self, image_path, image_paths, vectors):
    """
    Creates an embedding for the given image.
    Args:
      image_path (str): Path to the image file.
    """
    try:
      img = Image.open(image_path).convert("RGB")
      emb = model.encode(img, convert_to_tensor=False, show_progress_bar=False)
      image_paths.append(image_path)
      vectors.append(emb)
    except Exception as e:
      print(f"Error processing image {image_path}: {e}")

  def _initialise_model(self):
    """
    Initialise the embedding model if not already done.
    """
    global model
    if model is None:
      model = SentenceTransformer(MODEL_NAME)