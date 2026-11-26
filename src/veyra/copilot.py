"""Local retrieval only: fixed Sentence Transformer + aggregate-only Chroma index.

Manual indexing: python -m veyra.copilot index [--download-model]
The download flag retrieves public embedding weights only; no reports are sent.
Queries never download, rebuild, fit, persist query text, or load fraud models.
"""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
from uuid import uuid4

from .copilot_corpus import CorpusUnavailable, corpus, fingerprint, local_path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CONFIG = {"version": 1, "model": MODEL, "revision": REVISION, "tokens": 180,
          "overlap": 24, "distance": "cosine", "normalized": True}
COLLECTION = "veyra_aggregate_reports"
LOCK = Lock()


class CopilotUnavailable(ValueError):
    pass


class QueryTooLong(ValueError):
    pass


def storage(root):
    return local_path(root, ".veyra/copilot")


def encoder(root, download=False):
    # No telemetry, remote code, or credentials are needed for this public model.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, revision=REVISION, device="cpu", trust_remote_code=False,
                               local_files_only=not download, token=False,
                               cache_folder=str(local_path(root, ".veyra/copilot/embedding_cache")),
                               model_kwargs={"use_safetensors": True})


def client(root, index_id):
    if not re.fullmatch(r"[a-f0-9]{32}", index_id):
        raise CopilotUnavailable("Invalid local retrieval index; rebuild it manually.")
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    from chromadb import PersistentClient
    from chromadb.config import Settings
    return PersistentClient(path=str(local_path(root, f".veyra/copilot/indexes/{index_id}")),
                            settings=Settings(anonymized_telemetry=False, allow_reset=False))


def chunks(documents, model):
    output = []
    for document in documents:
        text = document["text"]
        prefix = f"{document['source']} | {document['section']}\n"
        # Budget includes source/section context and special tokens; never silently truncate.
        prefix_size = len(model.tokenizer.encode(prefix, add_special_tokens=True))
        budget = min(CONFIG["tokens"], model.max_seq_length - prefix_size - 4)
        if budget <= CONFIG["overlap"]:
            raise CopilotUnavailable("Source heading exceeds the embedding token budget.")
        offsets = model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
        for offset in range(0, len(offsets), budget - CONFIG["overlap"]):
            start, end = offsets[offset][0], offsets[min(offset + budget, len(offsets)) - 1][1]
            passage = text[start:end]
            embedded = prefix + passage
            if len(model.tokenizer.encode(embedded, add_special_tokens=True)) > model.max_seq_length:
                raise CopilotUnavailable("Chunk exceeds the embedding token budget.")
            metadata = {key: value for key, value in document.items() if key != "text"}
            metadata.update(char_start=start, char_end=end)
            output.append({"id": fingerprint({"metadata": metadata, "passage": passage}),
                           "text": passage, "embedded": embedded, "metadata": metadata})
            if offset + budget >= len(offsets):
                break
    if not output or len(output) > 10000:
        raise CopilotUnavailable("Corpus has an unsupported number of passages.")
    return output


def build(root, download=False):
    documents, manifest = corpus(root)
    model = encoder(root, download)
    passages = chunks(documents, model)
    vectors = model.encode([item["embedded"] for item in passages], batch_size=32,
                           normalize_embeddings=True, show_progress_bar=False)
    index_id = uuid4().hex
    db = client(root, index_id)
    collection = db.create_collection(COLLECTION, embedding_function=None,
                                      configuration={"hnsw": {"space": "cosine"}})
    for start in range(0, len(passages), 128):
        group = passages[start:start + 128]
        collection.add(ids=[item["id"] for item in group], documents=[item["text"] for item in group],
                       metadatas=[item["metadata"] for item in group], embeddings=vectors[start:start + 128].tolist())
    if collection.count() != len(passages) or corpus(root)[1] != manifest:
        raise CopilotUnavailable("Sources changed during indexing; the active index was not replaced.")
    manifest.update(index_id=index_id, config=CONFIG, chunks=len(passages),
                    created_at_utc=datetime.now(timezone.utc).isoformat())
    # Publish only a complete index. Prior index directories are never deleted automatically.
    active = storage(root) / "active.json"
    temporary = storage(root) / f"{index_id}.json.tmp"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    temporary.replace(active)
    return manifest


@lru_cache(maxsize=1)
def loaded(root, index_id):
    model = encoder(root)  # Offline only, even on the first API request.
    db = client(root, index_id)
    collection = db.get_collection(COLLECTION, embedding_function=None)
    return model, collection


def retrieve(root, query, top_k=5, min_similarity=0.2):
    """Bounded, read-only retrieval. Persisted data consists solely of approved passages."""
    if not LOCK.acquire(blocking=False):
        raise CopilotUnavailable("Copilot is busy. Retry shortly.")
    try:
        active = local_path(root, ".veyra/copilot/active.json")
        if not active.is_file():
            raise CopilotUnavailable("Copilot index unavailable. Run the manual indexing command first.")
        manifest = json.loads(active.read_text(encoding="utf-8"))
        documents, current = corpus(root)
        if manifest["config"] != CONFIG or manifest["corpus_sha256"] != current["corpus_sha256"]:
            raise CopilotUnavailable("Copilot index is stale. Review sources and rebuild the index manually.")
        index_id = manifest["index_id"]
        if not local_path(root, f".veyra/copilot/indexes/{index_id}/chroma.sqlite3").is_file():
            raise CopilotUnavailable("Copilot index unavailable. Rebuild it manually.")
        model, collection = loaded(root, index_id)
        if len(model.tokenizer.encode(query, add_special_tokens=True, verbose=False)) > model.max_seq_length:
            raise QueryTooLong("Query exceeds the embedding token limit. Please use a shorter question.")
        if collection.count() != manifest["chunks"]:
            raise CopilotUnavailable("Copilot index is incomplete. Rebuild it manually.")
        # Validate returned IDs against current approved chunks; never trust arbitrary DB text.
        approved = {item["id"]: item for item in chunks(documents, model)}
        vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        found = collection.query(query_embeddings=vector.tolist(), n_results=min(top_k, collection.count()), include=["distances"])
        matches = []
        for chunk_id, distance in zip(found["ids"][0], found["distances"][0]):
            if chunk_id not in approved or not math.isfinite(distance):
                raise CopilotUnavailable("Copilot index integrity check failed. Rebuild it manually.")
            similarity = max(-1.0, min(1.0, 1.0 - distance))
            if similarity < min_similarity:
                continue
            item = approved[chunk_id]
            matches.append({"passage": item["text"], "similarity": similarity,
                            "citation": {"chunk_id": chunk_id, **item["metadata"]}})
        return {"status": "ok" if matches else "no_matches", "matches": matches,
                "index": {"id": index_id, "created_at_utc": manifest["created_at_utc"],
                          "sources": len(current["sources"]), "chunks": manifest["chunks"],
                          "embedding_model": MODEL, "embedding_revision": REVISION},
                "skipped_sources": current["skipped"],
                "notice": "Retrieved report excerpts only, not an answer. Cosine similarity is not confidence or fraud probability. Reports describe their original analysis phase; later reports may supersede earlier statements. No query history is stored."}
    finally:
        LOCK.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["index"])
    parser.add_argument("--download-model", action="store_true", help="Allow one-time public embedding model download; never upload report text")
    args = parser.parse_args()
    try:
        manifest = build(Path(__file__).resolve().parents[2], args.download_model)
        print(f"Indexed {len(manifest['sources'])} approved sources / {manifest['chunks']} passages. Missing sources: {len(manifest['skipped'])}. Queries run offline.")
        return 0
    except (CorpusUnavailable, CopilotUnavailable) as exc:
        print(str(exc))
    except Exception:
        print("Index not published. Check optional Copilot dependencies, approved reports and cached embedding model. Use --download-model for initial setup; no report content is uploaded.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
