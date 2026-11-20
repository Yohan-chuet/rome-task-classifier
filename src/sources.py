import time

import pandas as pd
import src.constant as constant
from pathlib import Path


# methode to get all rome macros
def get_rome_macros():
    referentiel = pd.read_excel(constant.REFERENTIEL_ROME).drop_duplicates(subset=constant.COLONNES[0], keep="first")
    referentiel = referentiel[constant.COLONNES].reset_index(drop=True)
    return referentiel

# method to get all raw tasks for format task step
def raw_tasks():
    pass

def tasks_to_extract():
    return _get_task(constant.FILE_EXTRACT, column_name="extractions")

def save_extracted_task(df):
    _save_excel(df, constant.OUTPUT_DIR_EXTRACTED) 

def task_to_embed():
    return _get_task(constant.FILE_EMBEDDING)
    
# method to save ranking for embedding step
def save_embedding_results(df):
    macros = get_rome_macros().set_index(constant.COLONNES[0])
    df["macro_1_en"] = df["macro_1"].map(macros["libelle_macro_competence_en"])
    df["objectif_macro_1"] = df["macro_1"].map(macros["objectif"])
    df["domaine_macro_1"] = df["macro_1"].map(macros["domaine_competence"])
    _save_excel(df, constant.OUTPUT_DIR_EMBEDDING)



# method to get all tasks for qwen step
def task_for_qwen():
    pass

# method to save results for qwen step
def save_qwen_results():
    pass

def save_qwen_checkpoint():
    pass


def _save_excel(df, column):
    time_save = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(column)
    output_dir.mkdir(exist_ok=True)
    for part, start in enumerate(range(0, len(df), constant.EXCEL_MAX_ROWS), 1):
        path = output_dir / f"{time_save}_part{part}.xlsx"
        df.iloc[start:start + constant.EXCEL_MAX_ROWS].to_excel(path, index=False, engine="xlsxwriter")

def _get_task(files, column_name = "task"):
    dfs = []
    for file in files:
        df = pd.read_excel(file["path"], sheet_name=file["sheet"], engine="calamine")
        df = df.rename(columns={file["column"]: column_name})
        df[column_name] = df[column_name].fillna("").astype(str).str.strip()
        df["source_file"] = Path(file["path"]).name   
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)