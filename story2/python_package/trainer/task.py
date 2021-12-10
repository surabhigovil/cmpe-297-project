import argparse
import os
import git

# git.Git(".").clone("https://github.com/huggingface/transformers")
# os.system('pip install transformers')
import transformers
from transformers import Trainer
from trainer import experiment

def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """
    args_parser = argparse.ArgumentParser()

    # Saved model arguments
    args_parser.add_argument(
        '--job_dir',
        default="gs://output-story/models/story-gen",
        help='GCS location to export models')
    args_parser.add_argument(
        '--model_name',
        default="story-gen",
        help='The name of your saved model')
    args_parser.add_argument(
        '--batch_size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=4)
    args_parser.add_argument(
        '--num_epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --train-size and --num-epochs are specified,
        --train-steps will be: (train-size/train-batch-size) * num-epochs.\
        """,
        default=1,
        type=int,
    )
    args_parser.add_argument(
        '--learning_rate',
        help='Learning rate value for the optimizers.',
        default=1e-5,
        type=float)
    args_parser.add_argument(
        '--weight_decay',
        default=0.01,
        type=float)
    args_parser.add_argument(
        '--warmup',
        help='warmup ratio',
        default=0.1,
        type=float)
    args_parser.add_argument(
        '--adam_epsilon',
        help='adam epsilon',
        default=1e-8,
        type=float)

    return args_parser.parse_args()


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    print(args)
    experiment.run(args)


if __name__ == '__main__':
    main()

