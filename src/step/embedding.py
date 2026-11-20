import pandas as pd
import torch

import src.constant as constant
import src.sources as sources
from src.models.embedding_model import EmbeddingModel
from src.step.step import Step


class Embedding(Step):
    name = "Embedding"

    def process(self):
        self.model = EmbeddingModel(constant.MODEL["model_name"])
        self.labels = sources.get_rome_macros()[constant.COLONNES[0]].values
        self.macro_vectors = self.model.encode_macros(self.labels.tolist())

        df = sources.task_to_embed()
        to_classify = [t for t in df["task"].unique() if t]
        results = self.classify(to_classify)
        sources.save_embedding_results(df.merge(results, on="task", how="left"))

    def classify(self, tasks):
        """Découpe en blocs pour ne pas saturer la carte, puis assemble."""
        blocks = [self.classify_block(tasks[i:i + constant.BLOC])
                  for i in range(0, len(tasks), constant.BLOC)]
        return pd.concat(blocks, ignore_index=True)

    def classify_block(self, tasks):
        """Les TOP_K macros les plus proches de chaque tâche, avec leur score."""
        vectors = self.model.encode_tasks(tasks)
        scores, indices = torch.topk(vectors @ self.macro_vectors.T, k=constant.TOP_K)
        scores, indices = scores.cpu().numpy(), indices.cpu().numpy()

        result = pd.DataFrame({"task": tasks})
        for k in range(constant.TOP_K):
            result[f"macro_{k + 1}"] = self.labels[indices[:, k]]
            result[f"score_{k + 1}"] = scores[:, k].round(4)
        return result