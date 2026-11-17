import pandas as pd
import constant


# methode to get all rome macros
def rome_macros():
    referentiel = pd.read_excel(constant.REFERENTIEL_ROME).drop_duplicates(subset=constant.COLONNES, keep="first")
    referentiel = referentiel[constant.COLONNES].reset_index(drop=True)
    return referentiel


# method to get all raw tasks for format task step
def raw_tasks():
    pass

# method to get all tasks for embedding step
def task_to_embed():
    pass

# method to save ranking for embedding step
def save_embedding_results():
    pass

# method to get all tasks for qwen step
def task_for_qwen():
    pass

# method to save results for qwen step
def save_qwen_results():
    pass
