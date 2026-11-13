# method to get data from a source
TEST = False

REFERENTIEL_ROME = "path_to_referentiel_rome"


#//============Data for filter step============//
CHUNK_SIZE = 500_000
MIN_NBR_WORDS = 5
MIN_TASK_LENGTH = 2
data_to_filter = [
    dict(chemin="file_to_filter.xlsx", colonne="word", description="description"),
]


#//============Data for embedding step============//


