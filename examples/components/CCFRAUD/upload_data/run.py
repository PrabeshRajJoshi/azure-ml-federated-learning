"""Script for mock components."""
import argparse
import logging
import sys
import os

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from zipfile import ZipFile
from azureml.core import Run, Workspace
from azureml.core.keyvault import Keyvault

# helper with confidentiality
from confidential_io import EncryptedFile

def get_kaggle_client(kv: Keyvault):
    """Gets the Kaggle client

    Args:
        kv (Keyvault): keyvault to use for retrieving Kaggle credentials
    """

    os.environ["KAGGLE_USERNAME"] = kv.get_secret("kaggleusername")
    os.environ["KAGGLE_KEY"] = kv.get_secret("kagglekey")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def get_key_vault() -> Keyvault:
    """Retreives keyvault from current run"""
    run: Run = Run.get_context()
    logging.info(f"Got run context: {run}")
    workspace: Workspace = run.experiment.workspace
    return workspace.get_default_keyvault()


def download_kaggle_dataset(kaggle_client, path):
    """Downloads datasets to specified location

    Args:
        kaggle_client (KaggleApi): Instance of KaggleApi to use for retrieving the dataset
        path(str): location where to store downloaded dataset
    """
    kaggle_client.dataset_download_files("kartik2112/fraud-detection", path=path)


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """

    if args.silo_count < 1 or args.silo_count > 4:
        raise Exception("Number of splits/silos must be between 1 and 4 (included)!")

    kv = get_key_vault()
    kaggle_client = get_kaggle_client(kv)
    download_kaggle_dataset(kaggle_client, "./dataset")

    with ZipFile("./dataset/fraud-detection.zip", "r") as zObject:
        zObject.extractall("./dataset/extracted")

    df_train = pd.read_csv("./dataset/extracted/fraudTrain.csv", index_col=0)
    print(f"Loaded train dataset with {len(df_train)} rows")
    df_test = pd.read_csv("./dataset/extracted/fraudTest.csv", index_col=0)
    print(f"Loaded test dataset with {len(df_train)} rows")
    
    train_path = f"{args.raw_train_data}/raw.csv"
    test_path = f"{args.raw_test_data}/raw.csv"

    # draw random samples from a certain fraction of the total data
    sample_fraction = 1.0/args.silo_count * 0.7
    sample_random_state = int(str(args.silo_index)+str(args.silo_count))
    train_data_sampled = df_train.sample(frac=sample_fraction, random_state=sample_random_state)
    test_data_sampled = df_train.sample(frac=sample_fraction, random_state=sample_random_state)
    print(f"Sampled train dataset has {len(train_data_sampled)} rows")
    print(f"Sampled test dataset has {len(test_data_sampled)} rows")

    with EncryptedFile(train_path, "tw") as train_file:
        train_data_sampled.to_csv(train_file, index=False)
    with EncryptedFile(test_path, "tw") as test_file:
        test_data_sampled.to_csv(test_file, index=False)


def get_arg_parser(parser=None):
    """Parse the command line arguments for merge using argparse.

    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser):
        an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the component
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--silo_count",
        type=int,
        required=True,
        help="Number of silos",
    )
    parser.add_argument(
        "--silo_index",
        type=int,
        required=True,
        help="Index of the current silo",
    )
    parser.add_argument(
        "--raw_train_data",
        type=str,
        required=True,
        help="Output folder for train data",
    )
    parser.add_argument(
        "--raw_test_data",
        type=str,
        required=True,
        help="Output folder for test data",
    )
    return parser


def main(cli_args=None):
    """Component main function.

    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """

    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)

    print(f"Running script with arguments: {args}")
    run(args)


if __name__ == "__main__":
    # Set logging to sys.out
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    main()
