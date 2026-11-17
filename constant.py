# method to get data from a source
import os
from pathlib import Path


TEST = False
EXCEL_MAX_ROWS = 1_048_575

REFERENTIEL_ROME = "path_to_referentiel_rome"


#//============Data for filter step============//
CHUNK_SIZE = 500_000
MIN_NBR_WORDS = 5
MIN_TASK_LENGTH = 2
data_to_filter = [
    dict(chemin="file_to_filter.xlsx", colonne="word", description="description"),
]


#//============Data for embedding step============//

MODEL = dict(
    model_name="./models/nvidia_top_score/Nemotron-3-Embed-8B-BF16_bs64/checkpoint-84",  # ex model_2 (meilleur @1)
    prefix_document="passage: ",
    prefix_query="query: ",
)
FILE_EMBEDDING = [
    # file path, column holding the tasks, Excel sheet (name or number)
    dict(path="./data_a_classer/taches_generees_10_par_macro.xlsx", column="libelle_competence", sheet=0),
]
TOP_K = 10           # nombre de macros gardées par tâche
BLOC = 5000          # tâches encodées entre deux sauvegardes
BATCH_ENCODE = 32    # taille des lots sur la carte (32 laisse de la marge avec un modèle 8B)
OUTPUT_DIR_EMBEDDING = "embedding"

COLONNES = ["libelle_macro_competence", "libelle_macro_competence_en", "objectif", "domaine_competence"]

#//============Data for qwen step============//
FILES_QWEN = [
    # output of classer_embedding.py, task column, context column shown to Qwen (None = no context)
    dict(path="echantillon_100_par_macro.xlsx", column="word", description="description"),
    # dict(path="./classes_embedding/large_data_2026-11-09-10-03-23_part1_classe.xlsx", column="word", description="description"),
    # dict(path="./classes_embedding/large_data_2026-11-09-10-03-23_part2_classe.xlsx", column="word", description="description"),
]
OUTPUT_DIR_QWEN = "classes_qwen"
CHECKPOINT_DIR = Path(OUTPUT_DIR_QWEN) / "checkpoints"

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = "qwen3.8:latest"
NUM_CTX = 8192 
THINKING = False      # True = Qwen thinks before answering (better but much slower)
MIN_GPU_SHARE = 0.95  # below this share of Qwen on the GPU, Ollama runs it partly on the CPU (far too slow)
CHECK_GPU_EVERY = 100 # tasks between two GPU checks / progress messages

# Question asked to Qwen (same as classer_macro/qwen.py, evaluated in Claude_AZ)
SYSTEM_PROMPT = """You classify a work TASK, extracted from a job posting, into the French ROME skills framework.
You are given a numbered list of candidate macro-competences (skill groups). Each line is `number. macro-competence  [objective it belongs to]`.

RULES
1. Classify the TASK itself. CONTEXT is an excerpt of the job posting the task comes from; use it only to disambiguate a vague task. Never classify the CONTEXT.
2. Choose only among the numbered candidates. Prefer the candidate whose scope matches the main action of the task; personal qualities and soft skills are valid macro-competences.
3. Give your 5 best candidates, most likely first.
4. Set "not_a_task" to true only if the TASK is not a work activity or skill at all (e.g. a working condition, a benefit, a company slogan, a meaningless fragment). Still give your 5 candidates.
5. Reply with JSON only:
{"choices": [7, 2, 11, 4, 1], "not_a_task": false}"""

USER_PROMPT = """CANDIDATES
{candidates}

TASK: {task}
CONTEXT: {context}"""