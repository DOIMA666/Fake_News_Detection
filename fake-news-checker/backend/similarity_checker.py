from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

class SimilarityChecker:
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
    
    def encode_text(self, text):
        return self.model.encode(text, convert_to_tensor=True)
    
    def calculate_similarity(self, text1, text2):
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)
        
        similarity = util.cos_sim(embedding1, embedding2)
        
        return float(similarity[0][0])
    
    def calculate_similarity_batch(self, query_text, reference_texts):
        
        query_embedding = self.encode_text(query_text)
        
        reference_embeddings = self.model.encode(reference_texts, convert_to_tensor=True)
        
        similarities = util.cos_sim(query_embedding, reference_embeddings)[0]
        
        results = []
        for idx, (text, sim) in enumerate(zip(reference_texts, similarities)):
            results.append({
                'text': text,
                'similarity': float(sim),
                'index': idx
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def semantic_search(self, query_text, reference_texts, top_k=5):
        results = self.calculate_similarity_batch(query_text, reference_texts)
        return results[:top_k]
    
    def calculate_detailed_similarity(self, user_text, reference_text):
        overall_sim = self.calculate_similarity(user_text, reference_text)
        
        user_sentences = [s.strip() for s in user_text.split('.') if s.strip()]
        ref_sentences = [s.strip() for s in reference_text.split('.') if s.strip()]
        
        sentence_similarities = []
        
        if user_sentences and ref_sentences:
            user_embeddings = self.model.encode(user_sentences, convert_to_tensor=True)
            ref_embeddings = self.model.encode(ref_sentences, convert_to_tensor=True)
            
            for user_emb in user_embeddings:
                sims = util.cos_sim(user_emb, ref_embeddings)[0]
                max_sim = float(torch.max(sims))
                sentence_similarities.append(max_sim)
        
        avg_sentence_sim = np.mean(sentence_similarities) if sentence_similarities else 0
        min_sentence_sim = np.min(sentence_similarities) if sentence_similarities else 0
        max_sentence_sim = np.max(sentence_similarities) if sentence_similarities else 0
        
        return {
            'overall_similarity': overall_sim,
            'sentence_level': {
                'average': avg_sentence_sim,
                'minimum': min_sentence_sim,
                'maximum': max_sentence_sim,
                'count': len(sentence_similarities)
            }
        }
    
    def generate_verdict(self, similarity_score):
        if similarity_score >= 0.85:
            verdict = "HIGHLY_LIKELY_TRUE"
            label = "Rất có khả năng đúng"
            explanation = "Nội dung có độ tương đồng rất cao với các nguồn tin uy tín."
            color = "green"
        elif similarity_score >= 0.70:
            verdict = "LIKELY_TRUE"
            label = "Có khả năng đúng"
            explanation = "Nội dung khá tương đồng với các nguồn tin uy tín, nhưng cần xem xét thêm."
            color = "lightgreen"
        elif similarity_score >= 0.50:
            verdict = "UNCERTAIN"
            label = "Không chắc chắn"
            explanation = "Nội dung có một số điểm tương đồng nhưng cần kiểm chứng kỹ hơn."
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
        
        return {
            'verdict': verdict,
            'label': label,
            'explanation': explanation,
            'color': color,
            'similarity_score': similarity_score,
            'confidence': abs(similarity_score - 0.5) * 2  # 0 to 1
        }

if __name__ == "__main__":
    pass