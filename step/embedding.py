from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from pathlib import Path

import constant

class Embedding:
    def __init__(self):
        self.step_name = "Embedding"
        self.modele = None
        self.referentiel = None
        self.vecteurs_macros = None

    def run(self):
        print(f"Running step: {self.step_name}")
        Path(constant.OUTPUT_DIR).mkdir(exist_ok=True)
        self.load_model()
        self.load_reference()
        self.encode_macros()
        for file in constant.FILE_EMBEDDING:
            self.classify_file(file["path"], file["column"], file["sheet"])


    def load_model(self):
        self.modele = SentenceTransformer(constant.MODEL["model_name"])

    def load_reference(self):
        self.referentiel = pd.read_excel(constant.REFERENTIEL_ROME).drop_duplicates(subset=constant.COLONNES, keep="first")
        self.referentiel = self.referentiel[constant.COLONNES].reset_index(drop=True)

    def encode_macros(self):
        labels = self.reference[constant.COLONNES[0]].tolist()
        self.macro_vectors = self.encode(labels, constant.MODEL["prefix_document"])

    
    def encoder(self, textes, prefixe):
        vecteurs = self.modele.encode([prefixe + t for t in textes], batch_size=constant.BATCH_ENCODE,
                                      normalize_embeddings=True, convert_to_tensor=True,
                                      show_progress_bar=False)
        return vecteurs.float()


    def classify_file(self, path, column, sheet):
        name = Path(path).stem
        checkpoint = Path(constant.OUTPUT_DIR) / f"{name}.in_progress.parquet"

        df = pd.read_excel(path, sheet_name=sheet, engine="calamine")
        tasks = df[column].fillna("").astype(str).str.strip()
        already_done = self.load_checkpoint(checkpoint)
        to_classify = self.tasks_to_classify(tasks, already_done)

        results = self.classify_in_chunks(to_classify, already_done, checkpoint)
        results = self.add_macro_1_info(results)
        output = pd.concat([df, results.set_index("task").reindex(tasks.values).set_axis(df.index)], axis=1)
        
        self.write(output, name) # modify when db 
        checkpoint.unlink(missing_ok=True) 

    @staticmethod
    def load_checkpoint(checkpoint):
        """Tasks already classified during a previous run (empty if there are none)."""
        if checkpoint.exists():
            return pd.read_parquet(checkpoint)
        return pd.DataFrame({"task": []})

    @staticmethod
    def tasks_to_classify(tasks, already_done):
        """Unique, non-empty tasks that are not classified yet."""
        done = set(already_done["task"])
        return [t for t in pd.unique(tasks) if t and t not in done]

    def classify_in_chunks(self, tasks, already_done, checkpoint):
        results = already_done
        for start in range(0, len(tasks), constant.CHUNK_SIZE):
            chunk = tasks[start:start + constant.CHUNK_SIZE] 
            results = pd.concat([results, self.classify_chunk(chunk)])
            results.to_parquet(checkpoint, index=False)
            print(f"  {start + len(chunk)}/{len(tasks)} tasks classified")
        return pd.concat([results], ignore_index=True)

    def classify_chunk(self, chunk):
        """The TOP_K closest macros for each task of the chunk, with their score."""
        vectors = self.encode(chunk, constant.MODEL["prefix_query"])
        scores, indices = torch.topk(vectors @ self.macro_vectors.T, k=constant.TOP_K)
        scores, indices = scores.cpu().numpy(), indices.cpu().numpy()

        labels = self.reference[constant.COLUMN_NAME[0]].values
        result = pd.DataFrame({"task": chunk})
        for k in range(constant.TOP_K):
            result[f"macro_{k + 1}"] = labels[indices[:, k]]
            result[f"score_{k + 1}"] = scores[:, k].round(4)
        return result

    def add_macro_1_info(self, results):
        info = self.reference.set_index("libelle_macro_competence")
        results["macro_1_en"] = results["macro_1"].map(info["libelle_macro_competence_en"])
        results["objectif_macro_1"] = results["macro_1"].map(info["objectif"])
        results["domaine_macro_1"] = results["macro_1"].map(info["domaine_competence"])
        return results
    
    @staticmethod
    def write(output, name):
        if len(output) <= constant.EXCEL_MAX_ROWS:
            path = Path(constant.OUTPUT_DIR) / f"{name}_classe.xlsx"
            output.to_excel(path, index=False, engine="xlsxwriter")
        else:
            path = Path(constant.OUTPUT_DIR) / f"{name}_classe.csv"
            output.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  -> {path}")


