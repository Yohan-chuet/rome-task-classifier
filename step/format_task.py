import constant
import pandas as pd


class FormatTask:
    def __init__(self):
        self.step_name = "FormatTask"

    def run(self):
        print(f"Running step: {self.step_name}")
        # df_task = pd.read_excel(constant.data_to_filter[0]["chemin"])
        df_task = "not implemented"
        cleaned_task = self.clean_task(df_task)
        # filtered_task = self.filter_task(cleaned_task)

    def clean_task(self, df_task):
        print(f"{self.step_name} - Cleaning task: {df_task}")

    def filter_task(self, df_task):
        print(f"{self.step_name} - Filtering task: {df_task}")
        task = df_task[constant.data_to_filter[0]["colonne"]].fillna("")

        long_enough = task.str.len() >= constant.MIN_TASK_LENGTH
        enough_words = task.str.split().str.len() >= constant.MIN_NBR_WORDS
        # commence_par_verbe = task.apply(commence_par_un_verbe)

        to_keep = long_enough & enough_words # & commence_par_verbe

        return df_task[to_keep].reset_index(drop=True)

