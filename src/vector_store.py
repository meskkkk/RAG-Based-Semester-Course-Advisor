"""Embeddings and vector store (task B2).

Embedder
    Default: sentence-transformers ``all-MiniLM-L6-v2`` (downloaded once from
    Hugging Face, about 90 MB).  If that model cannot be loaded (no internet, the
    package is not installed) the code falls back to a TF-IDF embedder from
    scikit-learn so the system still runs.  Force one with the environment
    variable ``ADVISOR_EMBEDDER=minilm`` or ``ADVISOR_EMBEDDER=tfidf``.

Vector store
    FAISS ``IndexFlatIP`` over L2-normalised vectors (= cosine similarity).  If
    faiss is not installed, a NumPy matrix product gives the same results.

The index is saved in ``index/`` and reused.  It is rebuilt automatically when
any knowledge-base file changes (a hash of the files is stored with it) or when
a different embedder is used.

Public API:
    get_embedder(name=None) -> Embedder
    VectorStore.build(chunks, embedder) / .search(text, k, kind=None) / .save() / .load()
    get_store(rebuild=False) -> VectorStore      (cached, ready to search)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from chunker import Chunk, build_chunks
from kb_loader import KB_DIR

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

try:
    import faiss  # type: ignore
except Exception:      # pragma: no cover - depends on the machine
    faiss = None


# ------------------------------------------------------------------ embedders
class MiniLMEmbedder:
    """sentence-transformers model; vectors are L2-normalised."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = f"minilm:{model_name}"

    def fit(self, texts):
        return self            # pre-trained, nothing to fit

    def encode(self, texts) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


class TfidfEmbedder:
    """Offline fallback: TF-IDF (words + word pairs), fitted on the knowledge base."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
                                          stop_words="english", token_pattern=r"(?u)\b[A-Za-z0-9][A-Za-z0-9+#.]*\b")
        self.name = "tfidf"
        self._fitted = False

    def fit(self, texts):
        self.vectorizer.fit(list(texts))
        self._fitted = True
        return self

    def encode(self, texts) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fitted on the knowledge base first")
        m = self.vectorizer.transform(list(texts)).toarray().astype("float32")
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms


def get_embedder(name: str | None = None):
    """Return the requested embedder; fall back to TF-IDF if MiniLM cannot be loaded."""
    name = (name or os.environ.get("ADVISOR_EMBEDDER", "minilm")).lower()
    if name == "tfidf":
        return TfidfEmbedder()
    try:
        return MiniLMEmbedder(os.environ.get("ADVISOR_EMBED_MODEL", DEFAULT_MODEL))
    except Exception as exc:
        print(f"[vector_store] Could not load the sentence-transformers model ({type(exc).__name__}: "
              f"{str(exc)[:120]}). Falling back to TF-IDF embeddings.")
        return TfidfEmbedder()


# --------------------------------------------------------------- vector store
def kb_hash(kb_dir=None) -> str:
    kb_dir = Path(kb_dir) if kb_dir else KB_DIR
    h = hashlib.sha256()
    for name in ("courses.json", "course_descriptions.md", "registration_rules.md"):
        h.update(name.encode())
        h.update((kb_dir / name).read_bytes())
    return h.hexdigest()[:16]


class VectorStore:
    def __init__(self, chunks, vectors, embedder, kb_version=""):
        self.chunks = list(chunks)
        self.vectors = np.asarray(vectors, dtype="float32")
        self.embedder = embedder
        self.kb_version = kb_version
        self.backend = "faiss" if faiss is not None else "numpy"
        self._index = None
        if faiss is not None:
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, chunks=None, embedder=None, kb_dir=None) -> "VectorStore":
        chunks = chunks if chunks is not None else build_chunks(kb_dir)
        embedder = embedder or get_embedder()
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        return cls(chunks, embedder.encode(texts), embedder, kb_hash(kb_dir))

    # --------------------------------------------------------------- search
    def search(self, query: str, k: int = 5, kind: str | None = None) -> list:
        """Top-k chunks for a query: list of (Chunk, cosine score), best first.
        ``kind`` limits results to "course", "rule" or "terms" chunks."""
        q = self.embedder.encode([query])
        n = len(self.chunks)
        if self._index is not None:
            scores, idx = self._index.search(q, n)       # small corpus: score everything, then filter
            pairs = zip(idx[0].tolist(), scores[0].tolist())
        else:
            s = (self.vectors @ q[0]).tolist()
            pairs = sorted(enumerate(s), key=lambda t: -t[1])
        out = []
        for i, score in pairs:
            if i < 0:
                continue
            ch = self.chunks[i]
            if kind and ch.kind != kind:
                continue
            out.append((ch, float(score)))
            if len(out) >= k:
                break
        return out

    def scores_for(self, query: str, kind: str) -> dict:
        """Cosine score of every chunk of one kind, keyed by its ``ref`` (best chunk per ref)."""
        best = {}
        for ch, s in self.search(query, k=len(self.chunks), kind=kind):
            best[ch.ref] = max(s, best.get(ch.ref, -1.0))
        return best

    def get(self, ref: str) -> list:
        """All chunks for a course code or rule ID (exact lookup, no similarity)."""
        return [c for c in self.chunks if c.ref == ref]

    # ----------------------------------------------------------- persistence
    def save(self, directory=INDEX_DIR):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8")
        meta = {"embedder": self.embedder.name, "kb_version": self.kb_version,
                "n_chunks": len(self.chunks), "dim": int(self.vectors.shape[1])}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if isinstance(self.embedder, TfidfEmbedder):
            with open(d / "tfidf.pkl", "wb") as f:
                pickle.dump(self.embedder.vectorizer, f)
        if self._index is not None:
            faiss.write_index(self._index, str(d / "index.faiss"))

    @classmethod
    def load(cls, directory=INDEX_DIR, embedder=None) -> "VectorStore":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(x) for x in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
        vectors = np.load(d / "vectors.npy")
        if meta["embedder"] == "tfidf":
            embedder = TfidfEmbedder()
            with open(d / "tfidf.pkl", "rb") as f:
                embedder.vectorizer = pickle.load(f)
            embedder._fitted = True
        elif embedder is None:
            embedder = get_embedder()
        if embedder.name != meta["embedder"]:
            raise ValueError(f"index was built with {meta['embedder']}, current embedder is {embedder.name}")
        return cls(chunks, vectors, embedder, meta.get("kb_version", ""))


_STORE = None


def get_store(rebuild: bool = False, directory=INDEX_DIR, verbose: bool = True) -> VectorStore:
    """Load the saved index if it is up to date, otherwise build and save a new one."""
    global _STORE
    if _STORE is not None and not rebuild and _STORE.kb_version == kb_hash():
        return _STORE
    d = Path(directory)
    wanted = os.environ.get("ADVISOR_EMBEDDER", "minilm").lower()
    if not rebuild and (d / "meta.json").exists():
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            same_kind = (meta["embedder"] == "tfidf") == (wanted == "tfidf")
            if meta.get("kb_version") == kb_hash() and same_kind:
                _STORE = VectorStore.load(d)
                if verbose:
                    print(f"[vector_store] Loaded index: {len(_STORE.chunks)} chunks, "
                          f"embedder={_STORE.embedder.name}, backend={_STORE.backend}")
                return _STORE
        except Exception as exc:
            if verbose:
                print(f"[vector_store] Saved index not usable ({exc}); rebuilding.")
    _STORE = VectorStore.build()
    _STORE.save(d)
    if verbose:
        print(f"[vector_store] Built index: {len(_STORE.chunks)} chunks, "
              f"embedder={_STORE.embedder.name}, backend={_STORE.backend}, saved to {d}")
    return _STORE
