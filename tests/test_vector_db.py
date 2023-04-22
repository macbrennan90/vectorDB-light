import pytest
from vector_db import VectorDB
import numpy as np
import os
import json


class MockEmbeddingModel:
    vector_size = 5

    def embed(self, text: str) -> np.ndarray:
        # Set a seed for the random number generator based on the text's hash
        seed_value = hash(text) % 2**32
        np.random.seed(seed_value)

        # Generate a random embedding
        return np.random.rand(self.vector_size)


@pytest.fixture
def vector_db():
    embedding_model = MockEmbeddingModel()
    return VectorDB(embedding_model)


def test_add_document(vector_db):
    doc_id = vector_db.add_document("Document 1", {"metadata1": "value1"})
    assert len(vector_db.embeddings) == 1
    assert doc_id in vector_db.documents
    assert vector_db.documents[doc_id]["document"] == "Document 1"
    assert vector_db.documents[doc_id]["metadata"]["metadata1"] == "value1"


def test_query(vector_db):
    vector_db.add_document("Document 1", {"metadata1": "value1"})
    vector_db.add_document("Document 2", {"metadata2": "value2"})

    top_k_results = vector_db.query("query text", top_k=1)

    assert len(top_k_results) == 1
    assert "document" in top_k_results[0]
    assert "metadata" in top_k_results[0]
    assert "similarity" in top_k_results[0]


def test_load_documents_from_json(vector_db):
    data_directory = os.path.join(os.path.dirname(__file__), "..", "data")
    initial_documents_path = os.path.join(data_directory, "initial_documents.json")

    vector_db.load_documents_from_json(initial_documents_path)

    assert len(vector_db.embeddings) == 10
    assert len(vector_db.documents) == 10


def test_remove_document(vector_db):
    # Adding sample documents
    doc_1 = vector_db.add_document("New York City is bustling with people and traffic.", {"source": "observation"})
    doc_2 = vector_db.add_document("Central Park is a great place to relax and enjoy nature.", {"source": "observation"})
    doc_3 = vector_db.add_document("The subway system is an efficient way to travel.", {"source": "observation"})

    # Check initial state
    assert len(vector_db.embeddings) == 3
    assert len(vector_db.documents) == 3

    # Remove a document
    vector_db.remove_document(doc_2)

    # Check if the document was removed
    assert len(vector_db.embeddings) == 2
    assert len(vector_db.documents) == 2
    assert doc_2 not in vector_db.documents

    # Check if the embeddings and documents are in sync
    for doc_id, doc in vector_db.documents.items():
        assert np.allclose(vector_db.embeddings[doc['emb_idx']], vector_db._normalize_vector(vector_db.embedding_model.embed(doc["document"])))

    # Check if the remaining documents have the correct IDs
    assert vector_db.documents[doc_1]["document"] == "New York City is bustling with people and traffic."
    assert vector_db.documents[doc_3]["document"] == "The subway system is an efficient way to travel."
