from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer, util


class SimilarityChecker:
    def __init__(
        self, model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"
    ):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")

    def encode_text(self, text: str) -> Any:
        return self.model.encode(text, convert_to_tensor=True)

    def calculate_similarity_batch(
        self, query_text: str, reference_texts: List[str]
    ) -> List[Dict[str, Any]]:
        query_embedding = self.encode_text(query_text)

        reference_embeddings = self.model.encode(
            reference_texts, convert_to_tensor=True
        )

        similarities = util.cos_sim(query_embedding, reference_embeddings)[0]

        results = []
        for idx, (text, sim) in enumerate(zip(reference_texts, similarities)):
            results.append({"text": text, "similarity": float(sim), "index": idx})

        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results


if __name__ == "__main__":
    pass
