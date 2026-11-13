# import
import argparse


# step 
from step.embedding import Embedding
from step.extraction import Extraction

# Parameters
parser = argparse.ArgumentParser(
    description="Description de votre script"
)
parser.add_argument("-em", "--embedding", type=bool, default=False, help="use embedding step")
parser.add_argument("-ex", "--extraction", type=bool, default=False, help="use extraction step" )


def run_steps(steps):
    for step in steps:
        step.run()

if __name__ == "__main__":
    args = parser.parse_args()
    steps = []
    if args.embedding:
        steps.append(Embedding())
    if args.extraction:
        steps.append(Extraction())

    run_steps(steps)

    