import os
import argparse
import logging
import sys
import numpy as np
from typing import List

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import mlflow
import joblib
import json

# helper with confidentiality
from confidential_io import EncryptedFile

CATEGORICAL_PROPS = ["category", "region", "gender", "state"]
NUMERICAL_PROPS = [
        "age",
        "merch_lat",
        "merch_long",
        "lat",
        "long",
        "city_pop",
        "trans_date_trans_time",
        "amt",
    ]
USEFUL_PROPS = [
        "amt",
        "age",
        "merch_lat",
        "merch_long",
        "category",
        "region",
        "gender",
        "state",
        "lat",
        "long",
        "city_pop",
        "trans_date_trans_time",
    ]
TARGET_COLUMN = "is_fraud"


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

    parser.add_argument("--raw_data_dir", type=str, required=True, help="")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="")
    parser.add_argument("--metrics_prefix", type=str, required=False, help="Metrics prefix")
    parser.add_argument("--data_transformers_dir", type=str, required=False, help="Transformations to preprocess categorical and numerical columns in train/test/inference data")
    parser.add_argument("--data_transformers_dir_out", type=str, required=False, help="Transformations to preprocess categorical and numerical columns in train/test/inference data")

    return parser


def map_and_filter_data(df, useful_props):
    regions_df = pd.read_csv("./us_regions.csv")
    state_region = {row.StateCode: row.Region for row in regions_df.itertuples()}
    logger.info(f"Loaded state/regions:\n {state_region}")

    # map the region information
    df.loc[:, "region"] = df["state"].map(state_region)
    # map the age information
    df.loc[:, "age"] = (pd.Timestamp.now() - pd.to_datetime(df["dob"])) // pd.Timedelta("365D")
    # map the transaction date and time information
    datetimes = ["trans_date_trans_time"]  # "dob"
    for column in datetimes:
        df.loc[:, column] = pd.to_datetime(df[column]).view("int64")

    # Filter only useful columns
    df = df[useful_props]

    return df

def fit_and_save_encoders(dict_categories, categorical_props, categorical_encoders_path):
    """Creates one-hot encodings for categorical data

    Args:
        df (pd.DataFrame): Pandas dataframe to use to provide us with all unique value for each categorical column
    """

    ENCODERS = {}

    for column in categorical_props:
        if column not in ENCODERS:
            print(f"Creating encoder for column: {column}")
            # Simply set all zeros if the category is unseen
            encoder = OneHotEncoder(handle_unknown="ignore")
            encoder.fit(np.array(dict_categories[column]).reshape(-1, 1))
            ENCODERS[column] = encoder
    
    # save the encoders
    print(f"Saving encoders at {categorical_encoders_path}")
    joblib.dump(ENCODERS, categorical_encoders_path)

    return ENCODERS

def fit_and_save_scalers(df, numerical_props, numerical_scalers_path):
    """Creates standard scalers for numerical data

    Args:
        df (pd.DataFrame): Pandas dataframe to use to provide the numerical columns for transformation
    """

    SCALERS = {}

    for column in numerical_props:
        if column not in SCALERS:
            print(f"Creating encoder for column: {column}")
            scaler = StandardScaler()
            scaler.fit(df[column].values.reshape(-1, 1))
            SCALERS[column] = scaler

    # save the scalers
    print(f"Saving scalers at {numerical_scalers_path}")
    joblib.dump(SCALERS, numerical_scalers_path)

    return SCALERS

def apply_transforms(df, transformers_dir, is_training):
    """Apply categorical one hot encoders and numerical scalers

    Args:
        df (pd.DataFrame): Pandas dataframe to apply transforms to
    """
    encoders_file = os.path.join(transformers_dir,"encoders.pkl")
    scalers_file = os.path.join(transformers_dir,"scalers.pkl")
    
    logger.info(f"Applying transformations on categorical data...")
    if not is_training and os.path.exists(encoders_file):
        # load categorical encoders and transform the test/inference data
        ENCODERS = joblib.load(encoders_file)
    else:
        # prepare categorical encoders from agreed upon categories data and save at the desired location
        with open('./categories.json') as f:
            dict_categories = json.load(f)

        os.makedirs(transformers_dir, exist_ok=True)
        ENCODERS = fit_and_save_encoders(dict_categories, CATEGORICAL_PROPS, encoders_file)

    for column in CATEGORICAL_PROPS:
        encoder = ENCODERS.get(column)
        encoded_data = encoder.transform(df[column].values.reshape(-1, 1)).toarray()
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=[
                column + "_" + "_".join(x.split("_")[1:])
                for x in encoder.get_feature_names_out()
            ],
        )
        encoded_df.index = df.index
        df = df.join(encoded_df).drop(column, axis=1)
    
    logger.info(f"Applying transformations on numerical data...")

    if not is_training and os.path.exists(scalers_file):
        # load continuous scalers to transform the test/inference data
        SCALERS = joblib.load(scalers_file)
    else:
        os.makedirs(transformers_dir, exist_ok=True)
        # prepare continuous scalers from training data and save at the desired location
        SCALERS = fit_and_save_scalers(df, NUMERICAL_PROPS, scalers_file)

    for column in NUMERICAL_PROPS:
        scaler = SCALERS.get(column)
        df.loc[:, column] = scaler.transform(df[column].values.reshape(-1, 1))

    return df

def preprocess_data(
    raw_data_dir,
    processed_data_dir="./",
    metrics_prefix="default-prefix",
    data_transformers_dir=None,
    is_training=False
):
    """Preprocess the raw_training_data and raw_testing_data and save the processed data to train_data_dir and test_data_dir.

    Args:
        raw_data_dir: Training data directory that need to be processed
        processed_data_dir: Train data directory where processed train data will be saved
        metrics_prefix: prefix to identify metrics logged from the preprocessing component.
        data_transformers_dir: Directory to save or load the data transformers required for preprocessing.
        is_training: Flag to identify if this component is used during either training or testing/inference process.
    Returns:
        None
    """

    logger.info(
        f"Raw Data dir path: {raw_data_dir}, Processed Data dir path: {processed_data_dir}"
    )

    logger.debug(f"Loading data...")
    with EncryptedFile(raw_data_dir + f"/raw.csv", mode="rt") as raw_f:
        raw_df = pd.read_csv(raw_f)

    if TARGET_COLUMN in raw_df.columns:
        features_df = raw_df.drop(columns=[TARGET_COLUMN])
        restore_target = True
    else:
        features_df = raw_df
        restore_target = False

    df_mapped = map_and_filter_data(features_df, useful_props=USEFUL_PROPS)
    transformed_data = apply_transforms(df_mapped, data_transformers_dir, is_training)

    if restore_target:
        target_df = raw_df[TARGET_COLUMN]
        transformed_data = transformed_data.join(target_df)

    logger.debug(f"Transformed data samples: {len(transformed_data)}")

    os.makedirs(processed_data_dir, exist_ok=True)
    
    transformed_data = transformed_data.sort_values(by="trans_date_trans_time")
    
    processed_data_file = os.path.join(processed_data_dir, "processed.csv")
    logger.info(f"Saving processed data to {processed_data_file}")
    with EncryptedFile(processed_data_file, mode="wt") as processed_f:
        transformed_data.to_csv(processed_f, index=False)
    
    if is_training:
        fraud_weight = raw_df["is_fraud"].value_counts()[0] / raw_df["is_fraud"].value_counts()[1]
        logger.debug(f"Fraud weight: {fraud_weight}")

        fraud_weight_file = os.path.join(processed_data_dir, "fraud_weight.txt")
        logger.info(f"Saving fraud weight information to {fraud_weight_file}")
        with EncryptedFile(fraud_weight_file, mode="wt") as fraud_f:
            np.savetxt(fraud_f, np.array([fraud_weight]))

    # Mlflow logging
    log_metadata(transformed_data, metrics_prefix)


def log_metadata(df, metrics_prefix):
    with mlflow.start_run() as mlflow_run:
        # get Mlflow client
        mlflow_client = mlflow.tracking.client.MlflowClient()
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
        logger.debug(f"Root runId: {root_run_id}")
        if root_run_id:
            mlflow_client.log_metric(
                run_id=root_run_id,
                key=f"{metrics_prefix}/Number of datapoints",
                value=f"{df.shape[0]}",
            )



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
    logger.info(f"Running script with arguments: {args}")

    def run():
        """Run script with arguments (the core of the component).

        Args:
            args (argparse.namespace): command line arguments provided to script
        """
        if args.data_transformers_dir:
            transformers_dir = args.data_transformers_dir
            is_training = False
        else:
            logger.info(f"Transformations will be learnt and saved in the preprocessing step")
            is_training = True
            transformers_dir = args.data_transformers_dir_out

        preprocess_data(
            args.raw_data_dir,
            args.processed_data_dir,
            args.metrics_prefix,
            transformers_dir,
            is_training
        )

    run()


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
