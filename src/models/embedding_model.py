import src.constant as constant
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, path):
        self.model = SentenceTransformer(str(path))

    def encode_tasks(self, tasks):
        return self._encode(tasks, constant.MODEL["prefix_query"])

    def encode_macros(self, labels):
        return self._encode(labels, constant.MODEL["prefix_document"])

    def _encode(self, texts, prefix):
        vectors = self.model.encode([prefix + t for t in texts], batch_size=constant.BATCH_ENCODE,
                                    normalize_embeddings=True, convert_to_tensor=True,
                                    show_progress_bar=False)
        return vectors.float()