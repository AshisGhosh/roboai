from sentence_transformers import SentenceTransformer

class HuggingFaceMXBaiEmbedLarge:
    def __init__(self):
        self.model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    
    def embed(self, text):
        return self.model.encode(text).tolist()