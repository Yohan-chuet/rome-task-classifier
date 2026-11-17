import requests

import constant

import ctypes
import os
import pandas as pd

class Qwen:
    def __init__(self):
        self.step_name = "Qwen"
        self.reference = None

    def run(self):
        pass
        print(f"Running step: {self.step_name}")
        self.prevent_sleep()
        self.load_reference()
        try:
            self.ask("TASK: ok\nCONTEXT: ok", max_tokens=1)
            for file in constant.FILES_QWEN:
                self.classify_file(file["path"], file["column"], file["description"])
        finally:
            requests.post(f"{constant.OLLAMA_URL}/api/generate", json={"model": constant.LLM_MODEL, "keep_alive": 0}, timeout=30)

    def prevent_sleep(self):
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

    def load_reference(self):
        """One row per macro, indexed by its FR label (the one in the macro_ columns)."""
        referentiel = pd.read_excel(constant.REFERENTIEL_ROME).drop_duplicates(subset="libelle_macro_competence")
        self.reference = referentiel[constant.COLONNES].set_index("libelle_macro_competence")


    
       


    


