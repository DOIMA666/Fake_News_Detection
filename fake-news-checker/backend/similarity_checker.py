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

    def calculate_similarity(self, text1: str, text2: str) -> float:
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)

        similarity = util.cos_sim(embedding1, embedding2)

        return float(similarity[0][0])

    def calculate_similarity_batch(
        self, query_text: str, reference_texts: List[str]
    ) -> List[Dict[str, Any]]:
        query_embedding = self.encode_text(query_text)

        reference_embeddings = self.model.encode(
            reference_texts, convert_to_tensor=True
        )

        # Tính toán cosine similarity (1-vs-N)
        similarities = util.cos_sim(query_embedding, reference_embeddings)[0]

        results = []
        for idx, (text, sim) in enumerate(zip(reference_texts, similarities)):
            results.append({"text": text, "similarity": float(sim), "index": idx})

        # Sắp xếp kết quả, điểm cao nhất lên đầu
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results

    def generate_verdict(self, similarity_score: float) -> Dict[str, Any]:
        if similarity_score >= 0.85:
            verdict = "HIGHLY_LIKELY_TRUE"
            label = "Rất có khả năng đúng"
            explanation = "Nội dung có độ tương đồng rất cao với các nguồn tin uy tín."
            color = "green"
        elif similarity_score >= 0.70:
            verdict = "LIKELY_TRUE"
            label = "Có khả năng đúng"
            explanation = (
                "Nội dung khá tương đồng với các nguồn tin uy tín, "
                "nhưng cần xem xét thêm."
            )
            color = "lightgreen"
        elif similarity_score >= 0.50:
            verdict = "UNCERTAIN"
            label = "Không chắc chắn"
            explanation = (
                "Nội dung có một số điểm tương đồng nhưng cần kiểm chứng kỹ hơn."
            )
            color = "orange"
        elif similarity_score >= 0.30:
            verdict = "LIKELY_FALSE"
            label = "Có khả năng sai"
            explanation = "Nội dung có ít điểm tương đồng với các nguồn tin uy tín."
            color = "coral"
        else:
            verdict = "HIGHLY_LIKELY_FALSE"
            label = "Rất có khả năng sai"
            explanation = "Nội dung có độ tương đồng rất thấp với các nguồn tin uy tín."
            color = "red"

        # Tính toán độ tin cậy dựa trên khoảng cách
        confidence = abs(similarity_score - 0.5) * 2

        return {
            "verdict": verdict,
            "label": label,
            "explanation": explanation,
            "color": color,
            "similarity_score": similarity_score,
            "confidence": confidence,
        }


if __name__ == "__main__":
    pass
