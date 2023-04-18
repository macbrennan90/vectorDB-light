import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Union
import json


class VectorDB:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.embeddings = np.empty((0, embedding_model.vector_size))
        self.documents = {}

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector)

    def add_document(self, doc_id: int, text: str, metadata: Dict[str, Union[str, int, float]]) -> None:
        embedding = self.embedding_model.embed(text)
        normalized_embedding = self._normalize_vector(embedding)
        self.embeddings = np.vstack([self.embeddings, normalized_embedding])
        self.documents[doc_id] = {
            "document": text,
            "metadata": metadata,
        }

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Union[str, int, float]]]:
        query_embedding = self.embedding_model.embed(query_text)
        normalized_query_embedding = self._normalize_vector(query_embedding)
        similarities = cosine_similarity(self.embeddings, normalized_query_embedding.reshape(1, -1))
        top_k_indices = similarities.squeeze().argsort()[-top_k:][::-1]
        
        top_k_docs = []
        for index in top_k_indices:
            doc = self.documents[index]
            similarity = similarities[index][0]
            result = {
                "document": doc["document"],
                "metadata": doc["metadata"],
                "similarity": similarity,
            }
            top_k_docs.append(result)

        return top_k_docs

    def load_documents_from_json(self, json_file_path: str) -> None:
        with open(json_file_path, 'r') as file:
            documents_data = json.load(file)

        for doc_data in documents_data:
            doc_id = doc_data['doc_id']
            text = doc_data['text']
            metadata = doc_data['metadata']
            self.add_document(doc_id, text, metadata)
