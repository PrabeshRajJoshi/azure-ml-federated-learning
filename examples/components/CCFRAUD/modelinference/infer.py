import argparse
import os
import glob
from pathlib import Path
import mlflow
import pandas as pd
import torch
import numpy as np


parser = argparse.ArgumentParser("score")
parser.add_argument("--registered_model", type=str, help="Path to the input model")
parser.add_argument("--preprocessed_data_dir", type=str, help="Path to the data to score")
parser.add_argument(
    "--score_mode",
    type=str,
    help="The scoring mode. Possible values are `append` or `prediction_only`.",
)
parser.add_argument("--scores_dir", type=str, help="Path of predictions")

args = parser.parse_args()

lines = [
    f"model path: {args.registered_model}",
    f"scoring mode: {args.score_mode}",
    f"input data path: {args.preprocessed_data_dir}",
    f"ouputs data path: {args.scores_dir}",
]

for line in lines:
    print(f"/t{line}")

print("Loading model")
model = mlflow.pytorch.load_model(args.registered_model)

print(f"Reading all the CSV files from path {args.preprocessed_data_dir}")
input_files = glob.glob(args.preprocessed_data_dir + "/*.csv")

for input_file in input_files:
    print(f"Working on file {input_file}")
    df = pd.read_csv(input_file)

    predictions = {}
    with torch.no_grad():
        for index_val in df.index:
            # TODO: create prediction API for each model
            row = torch.tensor([df.iloc[index_val,:].values], dtype=torch.float)
            prediction,_ = model(row)
            print(f"inferred value: {prediction.item()}")
            predictions[index_val] = int(np.round(prediction.item()))

    # create the dataframe from predictions.
    prediction_df = pd.DataFrame.from_dict(predictions, orient='index', columns=["prediction"])
    if args.score_mode == "append":
        df.join(prediction_df)
    else:
        df = prediction_df

    output_file_name = Path(input_file).stem
    output_file_path = os.path.join(args.scores_dir, output_file_name + ".csv")
    print(f"Writing file {output_file_path}")
    df.to_csv(output_file_path, index=False)