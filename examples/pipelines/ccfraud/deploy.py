"""Federated Learning Cross-Silo deployment pipeline for Credit Card Fraud example.

This script:
1) reads a config file in yaml specifying the location of test data and other required parameters,
2) reads the components from a given folder,
3) builds a flexible pipeline depending on the config,
4) creates a required deployment using the last registered model.
"""
import os
import argparse
import random
import string
import datetime
import webbrowser

# Azure ML sdk v2 imports
import azure
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.entities import Data
from azure.ai.ml.entities import BatchEndpoint, PipelineComponentBatchDeployment

# to handle yaml config easily
from omegaconf import OmegaConf

#to read informatio from json config file
import json
from azure.ai.ml.exceptions import ErrorCategory, ErrorTarget, ValidationException

ENDPOINT_NAME = "fl-ccfraud-endpoint-"+datetime.date.today().strftime("%Y-%m-%d")

############################
### CONFIGURE THE SCRIPT ###
############################

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--config",
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    help="path to a config yaml file",
)
parser.add_argument(
    "--offline",
    default=False,
    action="store_true",
    help="Sets flag to not submit the experiment to AzureML",
)

parser.add_argument(
    "--subscription_id",
    type=str,
    required=False,
    help="Subscription ID",
)
parser.add_argument(
    "--resource_group",
    type=str,
    required=False,
    help="Resource group name",
)

parser.add_argument(
    "--workspace_name",
    type=str,
    required=False,
    help="Workspace name",
)

parser.add_argument(
    "--wait",
    default=False,
    action="store_true",
    help="Wait for the pipeline to complete",
)

parser.add_argument(
    "--update_transformers",
    default=False,
    action="store_true",
    help="Sets flag to update the preprocessing data transformers in the workspace and use them.",
)

parser.add_argument(
    "--submit_job",
    default=False,
    action="store_true",
    help="Sets flag to submit the pipeline component job in the workspace.",
)

parser.add_argument(
    "--create_endpoint",
    default=False,
    action="store_true",
    help="Sets flag to create and endpoint from the pipeline component in the workspace.",
)

parser.add_argument(
    "--delete_endpoint",
    default=False,
    action="store_true",
    help="Sets flag to create and endpoint from the pipeline component in the workspace.",
)

args = parser.parse_args()

# load the config from a local yaml file
YAML_CONFIG = OmegaConf.load(args.config)

# path to the components
COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", "CCFRAUD"
)

# path to the config file
CONFIG_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", ".."
)

###########################
### CONNECT TO AZURE ML ###
###########################

def get_tenant_id(config_path: str="config.json") -> str:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Checking the keys in the config.json file to check for required parameters.
    if not all([k in config.keys() for k in ("subscription_id", "resource_group", "workspace_name", "tenant_id")]):
        msg = (
            "The config file found in: {} does not seem to contain the required "
            "parameters. Please make sure it contains your subscription_id, "
            "resource_group, workspace_name, and tenant_id."
        )
        raise ValidationException(
            message=msg.format(config_path),
            no_personal_data_message=msg.format("[config_path]"),
            target=ErrorTarget.GENERAL,
            error_category=ErrorCategory.USER_ERROR,
        )
    # get information from config.json
    tenant_id_from_config = config["tenant_id"]
    return tenant_id_from_config

def connect_to_aml():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        tenant_id = get_tenant_id(
            config_path=os.path.join(CONFIG_FOLDER, "config.json")
            )
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential does not work
        credential = InteractiveBrowserCredential(tenant_id=tenant_id)

    # Get a handle to workspace
    try:
        # tries to connect using cli args if provided else using config.yaml
        ML_CLIENT = MLClient(
            subscription_id=args.subscription_id or YAML_CONFIG.aml.subscription_id,
            resource_group_name=args.resource_group
            or YAML_CONFIG.aml.resource_group_name,
            workspace_name=args.workspace_name or YAML_CONFIG.aml.workspace_name,
            credential=credential,
        )

    except Exception as ex:
        print("Could not find either cli args or config.yaml.")
        # tries to connect using local config.json
        ML_CLIENT = MLClient.from_config(credential=credential)

    return ML_CLIENT


#############################################
### GET ML_CLIENT AND COMPUTE INFORMATION ###
#############################################

if not args.offline:
    ML_CLIENT = connect_to_aml()
    COMPUTE_SIZES = ML_CLIENT.compute.list_sizes()


def get_gpus_count(compute_name):
    if not args.offline:
        ws_compute = ML_CLIENT.compute.get(compute_name)
        if hasattr(ws_compute, "size"):
            silo_compute_size_name = ws_compute.size
            silo_compute_info = next(
                (
                    x
                    for x in COMPUTE_SIZES
                    if x.name.lower() == silo_compute_size_name.lower()
                ),
                None,
            )
            if silo_compute_info is not None and silo_compute_info.gpus >= 1:
                return silo_compute_info.gpus
    return 1


####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
preprocessing_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "preprocessing", "spec.yaml")
)
model_inference_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "modelinference", "spec.yaml")
)


########################
### BUILD A PIPELINE ###
########################


def custom_fl_data_path(
    datastore_name, output_name, unique_id="${{name}}", iteration_num=None
):
    """Produces a path to store the data during FL training.

    Args:
        datastore_name (str): name of the Azure ML datastore
        output_name (str): a name unique to this output
        unique_id (str): a unique id for the run (default: inject run id with ${{name}})
        iteration_num (str): an iteration number if relevant

    Returns:
        data_path (str): direct url to the data path to store the data
    """
    data_path = f"azureml://datastores/{datastore_name}/paths/federated_learning/{output_name}/{unique_id}/"
    if iteration_num:
        data_path += f"iteration_{iteration_num}/"

    return data_path


def getUniqueIdentifier(length=8):
    """Generates a random string and concatenates it with today's date

    Args:
        length (int): length of the random string (default: 8)

    """
    str = string.ascii_lowercase
    date = datetime.date.today().strftime("%Y_%m_%d_")
    return date + "".join(random.choice(str) for i in range(length))


pipeline_identifier = getUniqueIdentifier()


@pipeline(
    description=f'FL deployment pipeline and the unique identifier is "{pipeline_identifier}" that can help you to track files in the storage account.',
)
def fl_ccfraud_infer(input_data: Input(type=AssetTypes.URI_FOLDER)):
    ######################
    ### PRE-PROCESSING ###
    ######################

    # run a pre-processing step for the inference data
    orchestrator_config = YAML_CONFIG.federated_learning.orchestrator
    transformers_asset = ML_CLIENT.data.get("inference_transformers", label="latest")

    inference_preprocessing_step = preprocessing_component(
        raw_data=input_data,
        metrics_prefix="inference",
        data_transformers=Input(
            type=AssetTypes.URI_FILE,
            mode=InputOutputModes.DOWNLOAD,
            path=transformers_asset.id,
        )
    )
    # if confidentiality is enabled, add the keyvault and key name as environment variables
    if hasattr(YAML_CONFIG, "confidentiality"):
        inference_preprocessing_step.environment_variables = {
            "CONFIDENTIALITY_DISABLE": str(not YAML_CONFIG.confidentiality.enable),
            "CONFIDENTIALITY_KEYVAULT": YAML_CONFIG.confidentiality.keyvault,
            "CONFIDENTIALITY_KEY_NAME": YAML_CONFIG.confidentiality.key_name,
        }
    else:
        inference_preprocessing_step.environment_variables = {
            "CONFIDENTIALITY_DISABLE": "True",
        }

    # add a readable name to the step
    inference_preprocessing_step.name = f"inference_preprocessing"

    # make sure the compute corresponds to the orchestrator
    inference_preprocessing_step.compute = orchestrator_config.compute

    # assign instance type for AKS, if available
    if hasattr(orchestrator_config, "instance_type"):
        inference_preprocessing_step.resources = {
            "instance_type": orchestrator_config.instance_type
        }

    # make sure the data is written in the right datastore
    inference_preprocessing_step.outputs.processed_data = Output(
        type=AssetTypes.URI_FOLDER,
        mode=InputOutputModes.MOUNT,
        path=custom_fl_data_path(orchestrator_config.datastore, "processed_inference_data"),
    )
    inference_preprocessing_step.outputs.data_transformers_out = Output(
        type=AssetTypes.URI_FOLDER,
        mode=InputOutputModes.MOUNT,
        path=orchestrator_config.data_transformers.path,
    )
    
    #################
    ### INFERENCE ###
    #################

    registered_model = ML_CLIENT.models.get(YAML_CONFIG.training_parameters.model_name, label="latest")
    
    # predict using the trained model
    inference_predict_step = model_inference_component(
        # The preprocessed data
        preprocessed_data = inference_preprocessing_step.outputs.processed_data,
        # The final aggregated model
        registered_model=Input(
            type=AssetTypes.MLFLOW_MODEL,
            mode=InputOutputModes.DOWNLOAD,
            path=registered_model.id
        ),
        # The mode to prepare the scored data
        score_mode=YAML_CONFIG.inference_parameters.score_mode
    )

    # if confidentiality is enabled, add the keyvault and key name as environment variables
    if hasattr(YAML_CONFIG, "confidentiality"):
        inference_predict_step.environment_variables = {
            "CONFIDENTIALITY_DISABLE": str(not YAML_CONFIG.confidentiality.enable),
            "CONFIDENTIALITY_KEYVAULT": YAML_CONFIG.confidentiality.keyvault,
            "CONFIDENTIALITY_KEY_NAME": YAML_CONFIG.confidentiality.key_name,
        }
    else:
        inference_predict_step.environment_variables = {
            "CONFIDENTIALITY_DISABLE": "True",
        }
    
    # add a readable name to the step
    inference_predict_step.name = "inference_prediction"

    # this is done in the orchestrator compute
    inference_predict_step.compute = orchestrator_config.compute
    
    # assign instance type for AKS, if available
    if hasattr(orchestrator_config, "instance_type"):
        inference_predict_step.resources = {
            "instance_type": orchestrator_config.instance_type
        }

    inference_predict_step.outputs.scores = Output(
            type=AssetTypes.URI_FOLDER,
            mode="rw_mount",
            path=custom_fl_data_path(
                YAML_CONFIG.federated_learning.orchestrator.datastore,
                "model_scores"
            ),
        )
    
    return {"scores": inference_predict_step.outputs.scores}
    
    # return {"scores": inference_preprocessing_step.outputs.processed_data}
    
pipeline_job_infer = fl_ccfraud_infer(
    input_data= Input(
                    type=AssetTypes.URI_FOLDER,
                    mode=InputOutputModes.DOWNLOAD,
                    path=os.path.join(os.path.dirname(__file__),"sample_inference_data"),
                )
)

# Inspect built pipeline
print(pipeline_job_infer)

if not args.offline:
    if args.update_transformers:
        # register the transformers in the workspace for inference. TODO: replace this with more meaningful transformers
        data_asset_to_create = Data(
            path=YAML_CONFIG.federated_learning.orchestrator.data_transformers.path,
            type=AssetTypes.URI_FOLDER,
            description="Sample transformation to help with inference. This asset needs to be replaced with more meaningful transformers in the future.",
            name="inference_transformers"
        )
        transformers_data_asset = ML_CLIENT.data.create_or_update(data_asset_to_create)
        print(
                f"Data asset {transformers_data_asset.name} with Version {transformers_data_asset.version} is registered"
            )
    if args.submit_job:
        print("Submitting the deployment pipeline job to your AzureML workspace...")
        pipeline_job = ML_CLIENT.jobs.create_or_update(
            pipeline_job_infer, experiment_name="fl_demo_ccfraud"
        )

        print("The url to see your live job running is returned by the sdk:")
        print(pipeline_job.services["Studio"].endpoint)

        webbrowser.open(pipeline_job.services["Studio"].endpoint)
    
    if args.create_endpoint:
        # create an online endpoint
        endpoint = BatchEndpoint(
            name=ENDPOINT_NAME,
            description="Online endpoint to test federated model."
        )

        ML_CLIENT.batch_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint with name {endpoint.name} is available for deployment.")
        #print(f"Endpoint scoring URI {endpoint.scoring_uri} is ready.")

        # create the pipeline component to infer from raw data
        pipeline_component = ML_CLIENT.components.create_or_update(
            fl_ccfraud_infer().component
        )

        # create a blue deployment
        blue_deployment = PipelineComponentBatchDeployment(
            name="fl-cc-fraud-blue-deployment",
            endpoint_name=endpoint.name,
            component=pipeline_component,
            settings={"continue_on_step_failure": False, "default_compute": YAML_CONFIG.federated_learning.orchestrator.compute},
        )

        # deploy the model
        ML_CLIENT.batch_deployments.begin_create_or_update(blue_deployment).result()

        # update the endpoint
        endpoint = ML_CLIENT.batch_endpoints.get(ENDPOINT_NAME)
        endpoint.defaults.deployment_name = blue_deployment.name
        # blue deployment takes 100 traffic, can be adjusted when a green deployment is developed
        endpoint.traffic = {"blue": 100}
        ML_CLIENT.batch_endpoints.begin_create_or_update(endpoint).result()

        print(f"The default deployment is {endpoint.defaults.deployment_name}")

        if args.delete_endpoint:
            ML_CLIENT.online_endpoints.begin_delete(name=ENDPOINT_NAME)


else:
    print("The pipeline was NOT submitted, omit --offline to send it to AzureML.")


