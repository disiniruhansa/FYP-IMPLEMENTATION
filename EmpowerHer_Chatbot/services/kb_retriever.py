# services/kb_retriever.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_kb_text(text: str, max_sentences: int = 3) -> str:
    if not text:
        return ""

    # Remove very loud headings (ALL CAPS lines)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not (ln.isupper() and len(ln) > 8)]
    text = " ".join(lines)

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Keep only first N sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sentences[:max_sentences]).strip()

    return text

@dataclass
class KBHit:
    chunk: str
    source: str
    score: float


class KnowledgeBaseRetriever:
    """
    Retriever over local text files in kb/docs/.
    - Splits documents into chunks
    - Supports either TF-IDF or embedding-based similarity
    - Retrieves top-k most relevant chunks for a query
    """

    def __init__(
        self,
        docs_dir: str = "kb/docs",
        chunk_size: int = 450,
        min_score: float = 0.12,
        backend: str = "embedding",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.min_score = min_score
        self.backend = (backend or "tfidf").lower()
        self.embed_model_name = embed_model

        self.chunks: List[str] = []
        self.sources: List[str] = []

        # TF-IDF fields
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        # Embedding fields (lazy-init to avoid heavy imports if unused)
        self.embed_tokenizer = None
        self.embed_model = None
        self.chunk_embeddings = None
        self.device = "cpu"
        self._torch = None

        self._load_and_index()

    # ---------------------------
    # Indexing
    # ---------------------------
    def _load_and_index(self) -> None:
        texts = self._load_texts()
        self._build_chunks(texts)

        if self.backend == "embedding":
            try:
                self._init_embedding_backend()
                self._build_embedding_index()
                print(f"[KB] Indexed {len(self.sources)} chunks from {len(texts)} documents using embeddings ({self.embed_model_name}).")
                return
            except Exception as exc:
                # Fall back to TF-IDF if embeddings fail (e.g., model download blocked)
                print(f"[KB] Embedding backend failed ({exc}); falling back to TF-IDF.")
                self.backend = "tfidf"

        self._build_tfidf_index(texts)

    def _load_texts(self) -> List[Tuple[str, str]]:
        if not self.docs_dir.exists():
            raise FileNotFoundError(
                f"KB folder not found: {self.docs_dir}. Create kb/docs and add .txt files."
            )

        texts: List[Tuple[str, str]] = []
        for fp in sorted(self.docs_dir.glob("*.txt")):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            text = clean_kb_text(self._clean(text), max_sentences=9999)
            if text.strip():
                texts.append((text, fp.name))

        if not texts:
            raise ValueError(f"No .txt documents found in {self.docs_dir}.")

        return texts

    def _build_chunks(self, texts: List[Tuple[str, str]]) -> None:
        for text, name in texts:
            for chunk in self._chunk_text(text, self.chunk_size):
                self.chunks.append(chunk)
                self.sources.append(name)

    def _build_tfidf_index(self, texts: List[Tuple[str, str]]) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        print(f"[KB] Indexed {len(self.sources)} chunks from {len(texts)} documents using TF-IDF.")

    def _init_embedding_backend(self) -> None:
        import torch  # Imported here to avoid mandatory dependency when using TF-IDF
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
        self.embed_model = AutoModel.from_pretrained(self.embed_model_name)
        self.embed_model.to(self.device)
        self.embed_model.eval()

    def _encode_texts(self, texts: List[str]):
        torch = self._torch
        assert torch is not None
        assert self.embed_tokenizer is not None and self.embed_model is not None

        inputs = self.embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.embed_model(**inputs)

        hidden = outputs.last_hidden_state  # (batch, seq, dim)
        mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq, 1)
        masked_hidden = hidden * mask
        summed = masked_hidden.sum(dim=1)  # (batch, dim)
        counts = mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()

    def _build_embedding_index(self) -> None:
        self.chunk_embeddings = self._encode_texts(self.chunks)

    # ---------------------------
    # Retrieval
    # ---------------------------
    def search(self, query: str, top_k: int = 2) -> List[KBHit]:
        q = self._clean(query)
        if not q.strip():
            return []

        hits: List[KBHit] = []

        if self.backend == "embedding" and self.chunk_embeddings is not None:
            torch = self._torch
            assert torch is not None

            q_emb = self._encode_texts([q])[0].unsqueeze(0)  # (1, dim)
            sims = torch.matmul(q_emb, self.chunk_embeddings.T).squeeze(0)
            scores = sims.tolist()
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            for idx in ranked_idx[: max(top_k * 3, 10)]:
                score = float(scores[idx])
                if score < self.min_score:
                    continue
                chunk = clean_kb_text(self.chunks[idx].strip(), max_sentences=3)
                src = self.sources[idx]
                hits.append(KBHit(chunk=chunk, source=src, score=score))
                if len(hits) >= top_k:
                    break

        else:
            assert self.vectorizer is not None and self.tfidf_matrix is not None

            q_vec = self.vectorizer.transform([q])
            sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            ranked_idx = sims.argsort()[::-1]

            for idx in ranked_idx[: max(top_k * 3, 10)]:
                score = float(sims[idx])
                if score < self.min_score:
                    continue
                chunk = clean_kb_text(self.chunks[idx].strip(), max_sentences=3)
                src = self.sources[idx]
                hits.append(KBHit(chunk=chunk, source=src, score=score))
                if len(hits) >= top_k:
                    break

        # Debug: show which sources matched and their scores
        if hits:
            print("[KB][search] query=", repr(q), "->", ", ".join(f"{h.source} ({h.score:.3f})" for h in hits))
        else:
            print("[KB][search] query=", repr(q), "-> no hits above threshold")

        return hits

    # ---------------------------
    # Helpers
    # ---------------------------
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        # split by paragraphs first
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: List[str] = []
        buf = ""

        for p in paras:
            if len(buf) + len(p) + 1 <= chunk_size:
                buf = (buf + " " + p).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = p

        if buf:
            chunks.append(buf)

        return chunks

    def _clean(self, text: str) -> str:
        text = text.replace("\r", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()


