from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        """
        Convert list of texts to embeddings
        """
        return self.model.encode(texts, show_progress_bar=True)
