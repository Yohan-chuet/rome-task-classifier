import pandas as pd
from pathlib import Path
import ast

import src.constant as constant
import src.sources as sources
from src.step.step import Step

class Extract(Step):
    def __init__(self):
        self.name = "Extract"

    def process(self):
        df = sources.tasks_to_extract()
        df = df[df["extractions"] != "[]"].copy() 

        df["extractions"] = df["extractions"].apply(ast.literal_eval)
        print(len(df))
        df = df.explode("extractions").reset_index(drop=True)
        for cle in constant.EXTRACTION_KEY:
            df[cle] = df["extractions"].apply(
                lambda x: x.get(cle) if isinstance(x, dict) else None
            )
        df = df.dropna(subset=["word"])
        df[["char_start", "char_end"]] = df[["char_start", "char_end"]].astype("Int64")

        columns = constant.EXTRACTION_KEY + ["description", "source_file"]
        df_2 = df[[c for c in columns if c in df.columns]].rename(columns={"word": "task"})

        sources.save_extracted_task(df_2)
