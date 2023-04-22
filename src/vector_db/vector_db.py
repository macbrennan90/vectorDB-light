import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Union
import json
import uuid


class VectorDB:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.embeddings = np.empty((0, embedding_model.vector_size))
        self.documents = {}
        self.embedding_idx_to_doc_id = []

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector)

    def add_document(self, text: str, metadata: Dict[str, Union[str, int, float]]) -> str:
        embedding = self.embedding_model.embed(text)
        normalized_embedding = self._normalize_vector(embedding)
        self.embeddings = np.vstack([self.embeddings, normalized_embedding])
        
        # Generate a unique UUID
        doc_id = str(uuid.uuid4())
        while doc_id in self.documents:
            doc_id = str(uuid.uuid4())

        self.documents[doc_id] = {
            "document": text,
            "metadata": metadata,
            "emb_idx": len(self.embeddings) - 1,
        }
        
        self.embedding_idx_to_doc_id.append(doc_id)

        return doc_id
    
    def remove_document(self, doc_id: int) -> None:
        if doc_id not in self.documents:
            raise ValueError(f"Document ID {doc_id} not found. It may have been deleted already.")

        emb_idx = self.documents[doc_id]["emb_idx"]
        self.embeddings = np.delete(self.embeddings, emb_idx, axis=0)
        self.embedding_idx_to_doc_id.pop(emb_idx)

        # Update embedding indices for remaining documents
        for remaining_doc_id, doc in self.documents.items():
            if doc["emb_idx"] > emb_idx:
                doc["emb_idx"] -= 1

        del self.documents[doc_id]

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Union[str, int, float]]]:
        query_embedding = self.embedding_model.embed(query_text)
        normalized_query_embedding = self._normalize_vector(query_embedding)
        similarities = cosine_similarity(self.embeddings, normalized_query_embedding.reshape(1, -1))
        top_k_indices = similarities.squeeze().argsort()[-top_k:][::-1]

        top_k_docs = []
        for index in top_k_indices:
            doc_id = self.embedding_idx_to_doc_id[index]
            doc = self.documents[doc_id]
            similarity = similarities[index][0]
            result = {
                "document": doc["document"],
                "metadata": doc["metadata"],
                "similarity": similarity,
                "doc_id": doc_id,
            }
            top_k_docs.append(result)

        return top_k_docs

    def load_documents_from_json(self, json_file_path: str) -> None:
        with open(json_file_path, 'r') as file:
            documents_data = json.load(file)

        for doc_data in documents_data:
            text = doc_data['text']
            metadata = doc_data['metadata']
            self.add_document(text, metadata)
