import os
from pathlib import Path
import argparse
import logging
import sys

# to load the pytorch model
import torch

# to save the model as mlflow model
import mlflow

# to get model architecture
import models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

############################
### CONFIGURE THE SCRIPT ###
############################

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
        "--aggregated_output",
        type=str, 
        required=True, 
        help="Path of the checkpoint for final averaged model"
    )

    parser.add_argument(
        "--model_name",
        type=str, 
        required=True, 
        help="Name of the model trained in each silo (SimpleLinear, SimpleLSTM, or SimpleVAE)"
    )

    parser.add_argument(
        "--output_model",
        type=str, 
        required=True, 
        help="Path of output model"
    )

    return parser

def register_model(model_checkpoint_path: str, 
                   model_name: str,
                   output_path:str):
    """
    Save the averaged model so that it can be deployed.\n
    Args:
        model_checkpoint_path (str): path where the aggregated model information is saved
        model_name (str): name of the model used to train in silos
        output_path (str): path of the folder where the model artifact is located
    """

    mlflow.autolog(log_models=False)

    # get the final state dictionary and input dimension of the model
    final_model_info = torch.load(
        os.path.join(model_checkpoint_path, "model.pt"),
        map_location=device
    )
    final_state_dict = final_model_info["state_dict"]
    input_dim = final_model_info["input_dim"]

    # initialize the model architecture
    model_to_deploy = getattr(models, model_name)(input_dim).to(device)

    # Create the signature object
    final_model_signature = "assign value here"

    # load the state dictionary into the model
    model_to_deploy.load_state_dict(final_state_dict)

    # set the model in evaluation mode
    model_to_deploy.eval()

    # save the mlflow model artifact in the accessible workspace path
    mlflow.pytorch.save_model(
        model_to_deploy,
        output_path
    )

    # register the model as the model_name, and update the version if an older version of this model exists
    mlflow.pytorch.log_model(
        model_to_deploy,
        artifact_path=model_name,
        registered_model_name=model_name
    )

def main(cli_args=None):
    """Component main function.

    It parses arguments and executes register_model() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)

    print(f"Running script with arguments: {args}")

    # save and register the aggregated model
    register_model(
        args.aggregated_output,
        args.model_name,
        args.output_model
    )

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
