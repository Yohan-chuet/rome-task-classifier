# import
import argparse


# step 
from step.embedding import Embedding
from step.format_task import FormatTask
from step.qwen import Qwen

# Parameters
parser = argparse.ArgumentParser(
    description="Description de votre script"
)
parser.add_argument("-em", "--embedding", type=bool, default=False, help="use embedding step")
parser.add_argument("-ft", "--format_task", type=bool, default=False, help="use format task step" )
parser.add_argument("-qw", "--qwen", type=bool, default=False, help="use Qwen step" )


def run_steps(steps):
    for step in steps:
        step.run()

if __name__ == "__main__":
    args = parser.parse_args()
    steps = []
    if args.embedding:
        steps.append(Embedding())
    if args.format_task:
        steps.append(FormatTask())
    if args.qwen:
        steps.append(Qwen())
    run_steps(steps)

    