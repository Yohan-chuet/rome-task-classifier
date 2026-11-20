from abc import ABC, abstractmethod
import time

class Step(ABC):
    name = "step"

    def run(self):
        start = time.time()
        print(f"--- {self.name}")
        self.process()
        print(f"--- {self.name} terminé ({(time.time() - start) / 60:.1f} min)")

    @abstractmethod
    def process(self):
        """Le travail propre à l'étape."""