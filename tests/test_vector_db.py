import pytest
from vector_db import VectorDB
import numpy as np
import os
import json


class MockEmbeddingModel:
    vector_size = 5

    def embed(self, text: str) -> np.ndarray:
        return np.random.rand(self.vector_size)


@pytest.fixture
def vector_db():
    embedding_model = MockEmbeddingModel()
    return VectorDB(embedding_model)


def test_add_document(vector_db):
    vector_db.add_document(0, "Document 1", {"metadata1": "value1"})
    assert len(vector_db.embeddings) == 1
    assert 0 in vector_db.documents
    assert vector_db.documents[0]["document"] == "Document 1"
    assert vector_db.documents[0]["metadata"]["metadata1"] == "value1"


def test_query(vector_db):
    vector_db.add_document(0, "Document 1", {"metadata1": "value1"})
    vector_db.add_document(1, "Document 2", {"metadata2": "value2"})

    top_k_results = vector_db.query("query text", top_k=1)

    assert len(top_k_results) == 1
    assert "document" in top_k_results[0]
    assert "metadata" in top_k_results[0]
    assert "similarity" in top_k_results[0]


def test_load_documents_from_json(vector_db, tmp_path):
    documents = [
        {"doc_id": 0, "text": "Document 1", "metadata": {"metadata1": "value1"}},
        {"doc_id": 1, "text": "Document 2", "metadata": {"metadata2": "value2"}},
    ]

    json_file = tmp_path / "documents.json"
    json_file.write_text(json.dumps(documents))

    vector_db.load_documents_from_json(str(json_file))

    assert len(vector_db.embeddings) == 2
    assert 0 in vector_db.documents
    assert 1 in vector_db.documents
    assert vector_db.documents[0]["document"] == "Document 1"
    assert vector_db.documents[1]["document"] == "Document 2"


def test_load_documents_from_json(vector_db):
    data_directory = os.path.join(os.path.dirname(__file__), "..", "data")
    initial_documents_path = os.path.join(data_directory, "initial_documents.json")

    vector_db.load_documents_from_json(initial_documents_path)

    assert len(vector_db.embeddings) == 10
    assert all(doc_id in vector_db.documents for doc_id in range(10))
